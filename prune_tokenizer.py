"""
Identify single character tokens (containing only one Chinese character) in Qwen3-8B tokenizer
Generate use_token_ids list for token-free pruning
"""
import json
import os

from transformers import AutoTokenizer


def is_chinese_char(char):
    """Check if character is a Chinese character"""
    return '\u4e00' <= char <= '\u9fff'


def find_single_char_tokens(tokenizer, vocab_size=None):
    """
    Iterate through vocabulary to find all single character tokens (containing only one Chinese character)
    
    Args:
        tokenizer: Qwen3 tokenizer
        vocab_size: Vocabulary size, if None use tokenizer.vocab_size
    
    Returns:
        single_char_token_ids: List of single character token IDs
        single_char_tokens: List of single character token texts
    """
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size

    single_char_token_ids = []
    single_char_tokens = []

    print(f"Scanning vocabulary (total {vocab_size} tokens)...")

    for token_id in range(vocab_size):
        # Decode token
        token = tokenizer.decode([token_id])

        # Check if single character and Chinese character
        if len(token) == 1 and is_chinese_char(token):
            single_char_token_ids.append(token_id)
            single_char_tokens.append(token)

    print(f"Scan complete, found {len(single_char_token_ids)} single character tokens")
    return single_char_token_ids, single_char_tokens


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Identify single character tokens in Qwen3-8B tokenizer")
    parser.add_argument("--model_path", type=str, default=None, help="Local model path (default: ./models/Qwen3-8B)")
    args = parser.parse_args()

    # Set default model path
    if args.model_path is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(project_root, "models", "Qwen3-8B")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}. "
                                f"Please download Qwen3-8B to this directory first. "
                                f"See README.md for download instructions.")

    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Find all single character tokens
    single_char_token_ids, single_char_tokens = find_single_char_tokens(tokenizer)

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "single_char_tokens.json")

    result = {
        "model_path": args.model_path,
        "vocab_size": tokenizer.vocab_size,
        "single_char_token_count": len(single_char_token_ids),
        "use_token_ids": single_char_token_ids,
        "single_char_tokens": single_char_tokens
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Single character token count: {len(single_char_token_ids)}")
    print(f"Example tokens (first 8): {single_char_tokens[:8]}")


if __name__ == "__main__":
    main()
