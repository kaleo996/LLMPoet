"""
Identify single character tokens (containing only one Chinese character) in Qwen3-8B tokenizer
Generate use_token_ids list for token-free pruning
Also annotate each character with Pingshui rhyme tone and rhyme group info
"""
import json
import os
import opencc
from pingshui_rhyme import PingZeClassifier
from collections import defaultdict
from transformers import AutoTokenizer


def is_chinese_char(char):
    """Check if character is a Chinese character"""
    return "\u4e00" <= char <= "\u9fff"


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


def annotate_pingshui(single_char_tokens, single_char_token_ids):
    """
    Annotate single char tokens with Pingshui rhyme info (tone and rhyme groups).

    Strategy: The Pingshui dictionary only contains traditional characters. We first expand
    the Pingshui dictionary by converting each traditional character to its simplified form.
    This way, a simplified character like 发 inherits pronunciation info from all its
    traditional counterparts (發 and 髮). Then we look up Qwen vocab characters directly in
    the expanded dictionary.

    Args:
        single_char_tokens: list of single char strings
        single_char_token_ids: list of corresponding token IDs

    Returns:
        tone_index: {"ping": [token_id, ...], "ze": [token_id, ...]}
        rhyme_index: {"上平聲一東": [token_id, ...], ...}
    """
    converter = opencc.OpenCC("t2s")
    classifier = PingZeClassifier()

    # Build mapping from pingshui dict: traditional char -> [{"tone": ..., "rhyme_group": ...}, ...]
    char_to_info = defaultdict(list)
    d = classifier.ping_ze_dict
    for tone_key in ("ping", "ze"):
        for _, groups in d[tone_key].items():
            for group_name, chars_str in groups.items():
                chars_str = chars_str[0]
                for ch in chars_str:
                    char_to_info[ch].append({"tone": tone_key, "rhyme_group": group_name})

    print(f"Pingshui dict contains {len(char_to_info)} unique traditional characters")

    # Expand the dictionary with simplified characters using t2s conversion.
    # For each traditional char, convert to simplified;
    # if different, add the simplified char with the same pronunciation info.
    simplified_added = 0
    for trad_ch in list(char_to_info.keys()):
        simp_ch = converter.convert(trad_ch)
        if simp_ch != trad_ch:
            for entry in char_to_info[trad_ch]:
                if entry not in char_to_info[simp_ch]:
                    char_to_info[simp_ch].append(entry)
            simplified_added += 1

    print(f"Expanded dictionary with {simplified_added} traditional->simplified mappings, "
          f"now {len(char_to_info)} unique characters total")

    # Build tone and rhyme indexes for single character tokens according to the expanded dictionary.
    tone_index = {"ping": [], "ze": []}
    rhyme_index = {}
    found_count = 0

    for ch, token_id in zip(single_char_tokens, single_char_token_ids):
        info = char_to_info.get(ch)
        if info:
            found_count += 1
            tones_seen = set()
            rhymes_seen = set()
            for entry in info:
                tone = entry["tone"]
                rg = entry["rhyme_group"]
                if tone not in tones_seen:
                    tones_seen.add(tone)
                    tone_index[tone].append(token_id)
                if rg not in rhymes_seen:
                    rhymes_seen.add(rg)
                    if rg not in rhyme_index:
                        rhyme_index[rg] = []
                    rhyme_index[rg].append(token_id)

    print(f"Annotated {found_count}/{len(single_char_tokens)} chars with pingshui info")
    print(f"Ping tone chars: {len(tone_index['ping'])}, Ze tone chars: {len(tone_index['ze'])}")
    print(f"Rhyme groups: {len(rhyme_index)}")

    return tone_index, rhyme_index


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

    # Annotate with Pingshui rhyme info
    tone_index, rhyme_index = annotate_pingshui(single_char_tokens, single_char_token_ids)

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "single_char_tokens.json")

    result = {
        "model_path": args.model_path,
        "vocab_size": tokenizer.vocab_size,
        "single_char_token_count": len(single_char_token_ids),
        "use_token_ids": single_char_token_ids,
        "single_char_tokens": single_char_tokens,
        "tone_index": tone_index,
        "rhyme_index": rhyme_index,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Single character token count: {len(single_char_token_ids)}")
    print(f"Example tokens (first 8): {single_char_tokens[:8]}")


if __name__ == "__main__":
    main()
