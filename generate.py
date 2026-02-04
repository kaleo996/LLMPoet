"""
LLMPoet generation script
Supports poetry generation using token-free Qwen3-8B
"""
import torch
import json
import os
import re

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))

from transformers import AutoTokenizer, GenerationConfig
from token_free_model import TokenFreeQwen3ForCausalLM
from utils import masked_poem_dict, poetry_prompt_template


def load_token_free_model(model_path=None, config_path=None, device="cuda"):
    """
    Load token-free Qwen3 model from local path
    
    Args:
        model_path: Local model path (default: ./models/Qwen3-8B)
        config_path: Token-free config file path (contains use_token_ids, default: ./single_char_tokens.json)
        device: Device
    
    Returns:
        model: TokenFreeQwen3ForCausalLM model
        tokenizer: tokenizer
    """
    # Set default model path
    if model_path is None:
        model_path = os.path.join(current_dir, "models", "Qwen3-8B")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. "
                                f"Please download Qwen3-8B to this directory. "
                                f"See README.md for download instructions.")

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Set default config path
    if config_path is None:
        config_path = os.path.join(current_dir, "single_char_tokens.json")

    # Load use_token_ids (if provided)
    use_token_ids = None
    if config_path and os.path.exists(config_path):
        print(f"Loading token-free config: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            single_char_tokens_config = json.load(f)
            use_token_ids = single_char_tokens_config.get("use_token_ids")
            if use_token_ids:
                print(f"Loaded {len(use_token_ids)} single character token IDs")

    # Create token-free model
    print(f"Creating token-free model from: {model_path}")
    # Use device_map="auto" for multi-GPU, or device for single device
    if device == "cuda" and torch.cuda.is_available():
        device_map = "auto"  # Let transformers handle device placement
    else:
        device_map = device
    model = TokenFreeQwen3ForCausalLM.from_pretrained(model_path,
                                                      use_token_ids=use_token_ids,
                                                      torch_dtype=torch.bfloat16,
                                                      device_map=device_map,
                                                      trust_remote_code=True)

    model.eval()
    print("Model loaded successfully")

    return model, tokenizer


def parse_template(masked_poem):
    """
    Parse template to extract character counts and punctuation positions
    
    Args:
        masked_poem: Template string with <|extra_1|> markers
    
    Returns:
        segments: List of (char_count, punctuation) tuples
        total_chars: Total number of characters needed
    """
    segments = []

    # Find all sequences of one or more <|extra_1|> followed by punctuation or end
    _placeholder = '<|extra_1|>'
    pattern = r'(<\|extra_1\|>)+([，。、；]?)'

    matches = re.finditer(pattern, masked_poem)
    for match in matches:
        punct = match.group(2) or ''
        placeholder_len = len(match.group(0)) - len(punct)
        char_count = placeholder_len // len(_placeholder)
        segments.append((char_count, punct))

    total_chars = sum(seg[0] for seg in segments)
    return segments, total_chars


def generate_poem(model,
                  tokenizer,
                  user_prompt,
                  poem_type,
                  device,
                  max_input_length=250,
                  top_k=50,
                  top_p=0.95,
                  temperature=0.8):
    """
    Generate poetry with automatic punctuation insertion
    
    Args:
        model: TokenFreeQwen3ForCausalLM model
        tokenizer: tokenizer
        user_prompt: User input prompt (e.g., "Write a poem about spring")
        poem_type: Poetry type (e.g., "五言绝句", "七言律诗", etc.)
        max_input_length: Maximum input length
        top_k: top-k sampling
        top_p: top-p sampling
        temperature: Temperature
        do_sample: Whether to sample
    
    Returns:
        generated_poem: Generated poetry text
    """
    # Get mask template
    if poem_type not in masked_poem_dict:
        raise ValueError(f"Unsupported poetry type: {poem_type}. Supported types: {list(masked_poem_dict.keys())}")

    masked_poem = masked_poem_dict[poem_type]

    # Parse template to get character counts and punctuation
    segments, total_chars_needed = parse_template(masked_poem)

    # Build prompt (keep original template with punctuation for context)
    prompt = poetry_prompt_template.format_map({
        "user_prompt": user_prompt,
        "masked_poem": masked_poem,
        "poem_type": poem_type
    })

    # Encode input
    input_ids = tokenizer.encode(prompt,
                                 return_tensors="pt",
                                 truncation=True,
                                 max_length=max_input_length,
                                 add_special_tokens=False).to(device)

    with torch.no_grad():
        output_ids = model.generate_poem_guided(
            input_ids=input_ids,
            tokenizer=tokenizer,
            segments=segments,
            total_chars_needed=total_chars_needed,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    generated_poem = tokenizer.decode(output_ids, skip_special_tokens=False)

    return generated_poem


def main():
    """Main function: example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="LLMPoet poetry generation")
    parser.add_argument("--model_path", type=str, default=None, help="Local model path (default: ./models/Qwen3-8B)")
    parser.add_argument("--config_path",
                        type=str,
                        default=None,
                        help="Token-free config file path (default: ./single_char_tokens.json)")
    parser.add_argument("--user_prompt", type=str, default="Write a poem about spring", help="User prompt")
    parser.add_argument("--poem_type", type=str, default="五言绝句", help="Poetry type")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_token_free_model(model_path=args.model_path,
                                             config_path=args.config_path,
                                             device=args.device)

    # Generate poetry
    print(f"\nUser prompt: {args.user_prompt}")
    print(f"Poetry type: {args.poem_type}")
    print("-" * 50)

    generated_poem = generate_poem(model=model,
                                   tokenizer=tokenizer,
                                   user_prompt=args.user_prompt,
                                   poem_type=args.poem_type,
                                   device=args.device)

    print("\nGenerated poetry:")
    print(generated_poem)
    print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
