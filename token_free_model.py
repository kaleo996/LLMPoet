"""
Token-free Qwen3 model wrapper class
Modify forward method to only allow single character token (Chinese characters) output
Supports optional per-position Pingshui rhyme prosody constraints (tone + rhyme)
"""
import torch
import torch.nn.functional as F
from transformers import Qwen3ForCausalLM
from typing import Optional, List, Tuple, Dict, Any
from utils import refine_constraint, PING_RHYME_GROUP_NAMES


class TokenFreeQwen3ForCausalLM(Qwen3ForCausalLM):
    """
    Token-free version based on Qwen3ForCausalLM
    Only allows single character token output (implemented via logits mask)
    Optionally applies per-position prosody constraints (tone and rhyme).
    """

    def __init__(self, config, use_token_ids: Optional[List[int]] = None, **kwargs):
        """
        Args:
            config: Qwen3Config
            use_token_ids: List of allowed token IDs (single character tokens)
        """
        super().__init__(config, **kwargs)

        if use_token_ids is not None:
            self.use_token_ids = use_token_ids
            # Build logits mask: only positions corresponding to use_token_ids are True
            self.logits_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
            self.logits_mask[use_token_ids] = True
        else:
            self.use_token_ids = None
            self.logits_mask = None

        # Prosody masks
        self.ping_mask = None
        self.ze_mask = None
        self.rhyme_index: Optional[Dict[str, List[int]]] = None
        self.token_to_ping_rhyme_groups: Optional[Dict[int, List[str]]] = None

    def _build_prosody_masks(self, prosody_config: dict):
        """
        Build prosody masks from tone_index and rhyme_index.

        Args:
            prosody_config: dict with keys "tone_index" and "rhyme_index"
                tone_index: {"ping": [token_ids], "ze": [token_ids]}
                rhyme_index: {"上平聲一東": [token_ids], ...}
        """
        vocab_size = self.config.vocab_size
        tone_index = prosody_config.get("tone_index", {})
        rhyme_index = prosody_config.get("rhyme_index", {})

        # Ping / Ze boolean masks over full vocabulary
        self.ping_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self.ping_mask[tone_index["ping"]] = True

        self.ze_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self.ze_mask[tone_index["ze"]] = True

        # Store rhyme_index for on-demand mask construction
        self.rhyme_index = rhyme_index

        # Build reverse map: token_id -> list of ping-tone rhyme group names
        # (only 上平聲 and 下平聲 groups, used for determining rhyme in regulated verse)
        self.token_to_ping_rhyme_groups = {}
        for group_name, token_ids in rhyme_index.items():
            if group_name in PING_RHYME_GROUP_NAMES:
                for tid in token_ids:
                    if tid not in self.token_to_ping_rhyme_groups:
                        self.token_to_ping_rhyme_groups[tid] = []
                    self.token_to_ping_rhyme_groups[tid].append(group_name)

        ping_count = int(self.ping_mask.sum().item())
        ze_count = int(self.ze_mask.sum().item())
        print(f"Prosody masks built: {ping_count} ping tokens, {ze_count} ze tokens, {len(rhyme_index)} rhyme groups")

    def _build_position_mask(
        self,
        constraint: dict,
        current_rhyme_group: Optional[str],
        anti_rhyme_groups: Optional[set] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build a position-specific boolean mask based on tone and rhyme constraints.

        Args:
            constraint: {"tone": "ping"|"ze"|"*", "is_rhyme": bool, "is_anti_rhyme": bool (optional)}
            current_rhyme_group: The rhyme group already chosen for this poem (None if not yet determined).
            anti_rhyme_groups: Set of rhyme group names that must be avoided at rhyme
                positions (populated from 首句不入韵 - the first line's last character).

        Returns:
            Boolean tensor of shape (vocab_size,) or None if no extra constraint.
        """
        tone = constraint["tone"]
        is_rhyme = constraint["is_rhyme"]
        is_anti_rhyme = constraint.get("is_anti_rhyme", False)

        # Tone mask
        if tone == "ping":
            mask = self.ping_mask.clone()
        elif tone == "ze":
            mask = self.ze_mask.clone()
        else:
            mask = None  # No tone constraint (flexible position)

        # Rhyme mask, only when rhyme group is already determined
        if is_rhyme and current_rhyme_group is not None:
            vocab_size = self.config.vocab_size
            rhyme_mask = torch.zeros(vocab_size, dtype=torch.bool)
            if current_rhyme_group in self.rhyme_index:
                rhyme_ids = self.rhyme_index[current_rhyme_group]
                rhyme_mask[rhyme_ids] = True
            if mask is not None:
                mask = mask & rhyme_mask
            else:
                mask = rhyme_mask

        # Anti-rhyme mask for is_anti_rhyme position (首句不入韵):
        # When rhyme group is already known (user-specified), explicitly exclude it.
        # When rhyme group is not yet known, skip for now, and the constraint will be enforced
        # in reverse at subsequent rhyme positions via anti_rhyme_groups.
        if is_anti_rhyme and current_rhyme_group is not None:
            vocab_size = self.config.vocab_size
            anti_rhyme_mask = torch.ones(vocab_size, dtype=torch.bool)
            if current_rhyme_group in self.rhyme_index:
                rhyme_ids = self.rhyme_index[current_rhyme_group]
                anti_rhyme_mask[rhyme_ids] = False
            if mask is not None:
                mask = mask & anti_rhyme_mask
            else:
                mask = anti_rhyme_mask

        # Anti-rhyme enforcement at rhyme positions:
        # When the rhyme group is not yet determined but anti_rhyme_groups is populated
        # from a 首句不入韵 first line, exclude tokens belonging to those groups so
        # the auto-detected rhyme group won't collide with the first line's ending.
        if is_rhyme and current_rhyme_group is None and anti_rhyme_groups:
            vocab_size = self.config.vocab_size
            anti_mask = torch.ones(vocab_size, dtype=torch.bool)
            for group_name in anti_rhyme_groups:
                if group_name in self.rhyme_index:
                    anti_mask[self.rhyme_index[group_name]] = False
            if mask is not None:
                mask = mask & anti_mask
            else:
                mask = anti_mask

        return mask

    def _get_ping_rhyme_group(self, token_id: int) -> Optional[str]:
        """Return the first ping-tone rhyme group for token_id, or None."""
        if self.token_to_ping_rhyme_groups is not None:
            groups = self.token_to_ping_rhyme_groups.get(token_id, [])
            if groups:
                return groups[0]
        return None

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        use_token_ids: Optional[List[int]] = None,
                        prosody_config: Optional[dict] = None,
                        *model_args,
                        **kwargs):
        """
        Load from pretrained model and apply token-free pruning + optional prosody masks.

        Args:
            pretrained_model_name_or_path: Model path or name
            use_token_ids: List of allowed token IDs
            prosody_config: dict with "tone_index" and "rhyme_index" (optional)
            *model_args, **kwargs: Arguments passed to parent class
        """
        # Load model first
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Apply token-free pruning
        if use_token_ids is not None:
            model.use_token_ids = use_token_ids
            model.logits_mask = torch.zeros(model.config.vocab_size, dtype=torch.bool)
            model.logits_mask[use_token_ids] = True
        else:
            model.use_token_ids = None
            model.logits_mask = None

        # Build prosody masks
        if prosody_config is not None:
            model._build_prosody_masks(prosody_config)

        return model

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
        temperature: float,
        position_mask: Optional[torch.Tensor] = None,
    ) -> int:
        """Sample one token from logits, respecting base + position masks."""
        mask_value = torch.finfo(logits.dtype).min

        # Combine base logits_mask with optional position_mask
        combined_mask: Optional[torch.Tensor] = None
        if self.logits_mask is not None:
            combined_mask = self.logits_mask.to(logits.device)
        if position_mask is not None:
            pm = position_mask.to(logits.device)
            combined_mask = (combined_mask & pm) if combined_mask is not None else pm

        # Fallback: if combined mask blocks everything, relax to base mask only
        if combined_mask is not None and not combined_mask.any():
            if self.logits_mask is not None:
                combined_mask = self.logits_mask.to(logits.device)
            else:
                combined_mask = None

        if combined_mask is not None:
            logits = torch.where(combined_mask, logits, mask_value)

        logits = logits[0, -1, :].float()
        logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_logits, _ = torch.topk(logits, k)
            threshold = topk_logits[..., -1, None]
            logits = torch.where(logits < threshold, torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype), logits)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumsum > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()
        return token_id

    @torch.no_grad()
    def generate_poem_guided(
        self,
        input_ids: torch.LongTensor,
        tokenizer: Any,
        segments: List[Tuple[int, str]],
        total_chars_needed: int,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.8,
        position_constraints: Optional[List[Dict]] = None,
        rhyme_group: Optional[str] = None,
    ) -> List[int]:
        """
        Generate poetry one character at a time following the template.
        At each character slot call forward once, sample one token;
        after each segment append template punctuation and update KV cache.

        Args:
            position_constraints: optional list of per-position constraint dicts
                [{"tone": "ping"|"ze"|"*", "is_rhyme": bool, ...}, ...].
                May include `"gu_ping_watch_positions"` keys for dynamic 孤平 prevention.
                Length must equal total_chars_needed when provided.
            current_rhyme_group: optional rhyme group name to use for the poem
                (e.g., "上平聲一東"). If None, the rhyme group will be
                automatically determined from the first rhyme position.

        Returns:
            generated_ids: list of generated token IDs (including punctuation).
        """
        device = input_ids.device
        generated_ids: List[int] = []
        past_key_values = None  # KV cache
        cur_input_ids = input_ids
        segment_idx = 0
        char_in_segment = 0

        # Prosody state
        current_rhyme_group: Optional[str] = rhyme_group
        # Check if current_rhyme_group is valid
        if current_rhyme_group is not None:
            if current_rhyme_group not in PING_RHYME_GROUP_NAMES:
                raise ValueError(
                    f"Invalid rhyme group for regulated verse: {current_rhyme_group}. "
                    f"Valid groups: {PING_RHYME_GROUP_NAMES}"
                )
        global_char_idx = 0
        generated_tones: Dict[int, str] = {}  # idx -> "ping"|"ze" for 孤平 checks
        anti_rhyme_groups: set = set() # 首句不入韵, rhyme groups of 1st line's last char that subsequent rhyme positions must avoid

        # Generate total_chars_needed tokens
        for _ in range(total_chars_needed):
            outputs = self.forward(
                input_ids=cur_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits

            # Build position-specific mask
            position_mask = None
            constraint = None
            if (position_constraints is not None and global_char_idx < len(position_constraints)):
                constraint = position_constraints[global_char_idx]
                # Dynamically tighten constraint to prevent 孤平
                constraint = refine_constraint(constraint, generated_tones)
                position_mask = self._build_position_mask(constraint, current_rhyme_group, anti_rhyme_groups)

            token_id = self._sample_next_token(
                logits, top_k=top_k, top_p=top_p, temperature=temperature,
                position_mask=position_mask,
            )

            # Track generated tone for dynamic 孤平 prevention
            # Prioritize applied constraint tone for strict positions (handles polyphonic characters better)
            if constraint is not None and constraint["tone"] in ("ping", "ze"):
                generated_tones[global_char_idx] = constraint["tone"]
            elif self.ping_mask[token_id]:
                generated_tones[global_char_idx] = "ping"
            elif self.ze_mask[token_id]:
                generated_tones[global_char_idx] = "ze"

            # 首句不入韵
            if (constraint is not None
                    and constraint.get("is_anti_rhyme", False)
                    and current_rhyme_group is None):
                groups = self.token_to_ping_rhyme_groups.get(token_id, [])
                anti_rhyme_groups.update(groups)

            # Update rhyme group on first rhyme position
            if constraint is not None and constraint["is_rhyme"]:
                if current_rhyme_group is None:
                    current_rhyme_group = self._get_ping_rhyme_group(token_id)

            generated_ids.append(token_id)
            char_in_segment += 1
            global_char_idx += 1

            # If the current segment is complete, append punctuation
            if char_in_segment == segments[segment_idx][0]:
                punctuation = segments[segment_idx][1]
                if punctuation:
                    punct_ids = tokenizer.encode(punctuation, add_special_tokens=False)[0]
                    generated_ids.append(punct_ids)
                    cur_input_ids = torch.tensor([[token_id]], device=device, dtype=torch.long)
                    outputs = self.forward(
                        input_ids=cur_input_ids,
                        past_key_values=outputs.past_key_values,
                        use_cache=True,
                    )
                    token_id = punct_ids
                segment_idx += 1
                char_in_segment = 0

            cur_input_ids = torch.tensor([[token_id]], device=device, dtype=torch.long)
            past_key_values = outputs.past_key_values  # Update KV cache

        return generated_ids
