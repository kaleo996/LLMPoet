import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_CACHE: dict[tuple[str, str], tuple[AutoModelForCausalLM, AutoTokenizer]] = {}


def resolve_model_path(model_path: Optional[str] = None) -> str:
    if model_path is None:
        model_path = os.path.join(_PROJECT_ROOT, "ckpt", "Qwen3-8B")
    return model_path


def load_local_model(model_path: Optional[str] = None, device: str = "cuda"):
    resolved_path = resolve_model_path(model_path)
    cache_key = (resolved_path, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not os.path.isdir(resolved_path):
        raise FileNotFoundError(
            f"Model not found at {resolved_path}. Please download the local checkpoint first."
        )

    tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" and torch.cuda.is_available() else device,
        trust_remote_code=True,
    )
    model.eval()
    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    *,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
