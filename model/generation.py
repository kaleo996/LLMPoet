"""
Core poetry generation logic: load token-free model, generate_poem with structure from poem_type.
Used by both CLI (cli.py) and Web UI (app.py).
"""
import json
import os
import torch
from transformers import AutoTokenizer

from .token_free_model import TokenFreeQwen3ForCausalLM
from .utils import (
    POEM_STRUCTURE,
    get_poem_structure,
    get_segments,
    get_prompt_template,
    get_poem_type_display,
    get_position_constraints,
)

# Project root (parent of model/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_token_free_model(model_path=None, config_path=None, device="cuda"):
    """
    Load token-free Qwen3 model from local path.

    Args:
        model_path: Local model path (default: ./ckpt/Qwen3-8B)
        config_path: Token-free config file path (contains use_token_ids, default: ./ckpt/single_char_tokens.json)
        device: Device

    Returns:
        model: TokenFreeQwen3ForCausalLM model
        tokenizer: tokenizer
    """
    if model_path is None:
        model_path = os.path.join(_PROJECT_ROOT, "ckpt", "Qwen3-8B")
    if config_path is None:
        config_path = os.path.join(_PROJECT_ROOT, "ckpt", "single_char_tokens.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please download Qwen3-8B to this directory. "
            "See README.md for download instructions."
        )

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    use_token_ids = None
    prosody_config = None
    if config_path and os.path.exists(config_path):
        print(f"Loading token-free config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            single_char_tokens_config = json.load(f)
            use_token_ids = single_char_tokens_config.get("use_token_ids")
            tone_index = single_char_tokens_config.get("tone_index")
            rhyme_index = single_char_tokens_config.get("rhyme_index")
            prosody_config = {
                "tone_index": tone_index,
                "rhyme_index": rhyme_index,
            }
            print(
                f"Loaded {len(use_token_ids)} single character token IDs, "
                f"and {len(tone_index)} tone groups and {len(rhyme_index)} rhyme groups"
            )

    print(f"Creating token-free model from: {model_path}")
    if device == "cuda" and torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = device
    model = TokenFreeQwen3ForCausalLM.from_pretrained(
        model_path,
        use_token_ids=use_token_ids,
        prosody_config=prosody_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    model.eval()
    print("Model loaded successfully")

    return model, tokenizer


def generate_poem(
    model,
    tokenizer,
    user_prompt,
    poem_type,
    device,
    variant=None,
    rhyme_group=None,
    script="simplified",
    max_input_length=250,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
):
    """
    Generate poetry with automatic punctuation insertion.

    Args:
        model: TokenFreeQwen3ForCausalLM model
        tokenizer: tokenizer
        user_prompt: User input prompt (e.g., "Write a poem about spring")
        poem_type: Poetry type (e.g., "五言绝句", "七言律诗", etc.)
        variant: Metrical pattern variant name (e.g., "仄起首句不入韵"). None = random selection.
        rhyme_group: Rhyme group name (e.g., "上平聲一東"). None = auto-detect from first rhyme position.
        script: Output script, "simplified" or "traditional"
        max_input_length: Maximum input length
        top_k: top-k sampling
        top_p: top-p sampling
        temperature: Temperature

    Returns:
        generated_poem: Generated poetry text
    """
    if poem_type not in POEM_STRUCTURE:
        raise ValueError(
            f"Unsupported poetry type: {poem_type}. Supported types: {list(POEM_STRUCTURE.keys())}"
        )

    segments = get_segments(poem_type)
    total_chars_needed = sum(seg[0] for seg in segments)
    position_constraints, variant_name = get_position_constraints(poem_type, variant_name=variant)

    num_lines, chars_per_line = get_poem_structure(poem_type)
    prompt_template = get_prompt_template(script)
    display_type = get_poem_type_display(poem_type, script)
    prompt = prompt_template.format_map({
        "user_prompt": user_prompt,
        "poem_type": display_type,
        "num_lines": num_lines,
        "chars_per_line": chars_per_line,
    })

    input_ids = (
        tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )
        .to(device)
    )

    with torch.no_grad():
        output_ids = model.generate_poem_guided(
            input_ids=input_ids,
            tokenizer=tokenizer,
            segments=segments,
            total_chars_needed=total_chars_needed,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            position_constraints=position_constraints,
            rhyme_group=rhyme_group,
        )
    generated_poem = tokenizer.decode(output_ids, skip_special_tokens=False)

    return generated_poem
