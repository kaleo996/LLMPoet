"""
Token-free Qwen3 model wrapper class
Modify forward method to only allow single character token (Chinese characters) output
"""
import torch
from transformers import Qwen3ForCausalLM
from typing import Optional, Union, List, Tuple


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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        """
        Override forward method to apply mask after computing logits
        """
        # Call parent class forward
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Apply mask if logits_mask is set
        if self.logits_mask is not None and outputs.logits is not None:
            mask_value = torch.finfo(outputs.logits.dtype).min
            mask = self.logits_mask.to(outputs.logits.device)
            # Apply mask: set logits that don't meet requirements to negative infinity
            outputs.logits = torch.where(mask, outputs.logits, mask_value)

        return outputs
