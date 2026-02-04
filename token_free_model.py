"""
Token-free Qwen3 model wrapper class
Modify forward method to only allow single character token (Chinese characters) output
"""
import torch
import torch.nn.functional as F
from transformers import Qwen3ForCausalLM
from typing import Optional, List, Tuple, Any


class TokenFreeQwen3ForCausalLM(Qwen3ForCausalLM):
    """
    Token-free version based on Qwen3ForCausalLM
    Only allows single character token output (implemented via logits mask)
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

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        use_token_ids: Optional[List[int]] = None,
                        *model_args,
                        **kwargs):
        """
        Load from pretrained model and apply token-free pruning
        
        Args:
            pretrained_model_name_or_path: Model path or name
            use_token_ids: List of allowed token IDs
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

        return model

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> int:
        """Sample one token from logits."""
        # Apply mask to avoid generating tokens not allowed
        if self.logits_mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            mask = self.logits_mask.to(logits.device)
            logits = torch.where(mask, logits, mask_value)

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
    ) -> torch.LongTensor:
        """
        Generate poetry one character at a time following the template.
        At each character slot call forward once, sample one token;
        after each segment append template punctuation and update KV cache.
        Returns full sequence (input_ids + generated token ids) as (1, seq_len).
        """
        device = input_ids.device
        generated_ids: List[int] = []
        past_key_values = None # KV cache
        cur_input_ids = input_ids
        segment_idx = 0
        char_in_segment = 0

        # Generate total_chars_needed tokens
        for _ in range(total_chars_needed):
            outputs = self.forward(
                input_ids=cur_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits
            token_id = self._sample_next_token(
                logits, top_k=top_k, top_p=top_p, temperature=temperature
            )
            generated_ids.append(token_id)
            char_in_segment += 1

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
            past_key_values = outputs.past_key_values # Update KV cache

        return generated_ids
