"""
Extract poem themes using local Qwen3-8B.

Reads filtered_poems.json, runs inference with ckpt/Qwen3-8B for each poem
to get a short theme in traditional Chinese. Saves poem_themes.json (id -> theme)
for build_dataset.py. Supports --limit for testing and resume from existing theme cache.
"""

import os
import json
import re
import argparse
from typing import List, Dict, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# System prompt in traditional Chinese
THEME_SYSTEM_TC = "請根據詩的標題與正文，用一個繁體中文短句概括其主題，包括詩歌的意象、主旨等。只輸出主題本身，不要解釋、換行。"


def load_model_and_tokenizer(model_path: str, device: str):
    """Load Qwen3-8B and tokenizer from local path."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Place Qwen3-8B there or set --model_path.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" and torch.cuda.is_available() else device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _normalize_theme(raw: str) -> Optional[str]:
    raw = raw.strip()
    raw = re.sub(r"[\s\n]+", "", raw)
    idx = raw.find("主題：")
    if idx != -1:
        raw = raw[idx + len("主題："):]
    if raw and re.match(r".*[，。；：？！]$", raw):
        raw = raw[:-1]
    if not raw:
        return None
    return raw


def extract_themes_batch(
    model,
    tokenizer,
    poems_batch: List[Dict],
    max_new_tokens: int = 64,
    max_input_length: int = 1024,
) -> List[Optional[str]]:
    """Run batch inference; return list of theme strings (or None)."""
    if not poems_batch:
        return []
    texts = []
    for p in poems_batch:
        messages = [
            {"role": "system", "content": THEME_SYSTEM_TC},
            {"role": "user", "content": f"標題：{p['title']}\n\n正文：\n{p['poem_text']}"},
        ]
        t = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        texts.append(t)

    was_left = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_attention_mask=True,
        ).to(model.device)
    finally:
        tokenizer.padding_side = was_left

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # With left padding, prompt occupies the last input_lengths[i] positions of the
    # padded input; generated tokens start at index padded_len.
    padded_len = inputs.input_ids.shape[1]
    themes = []
    for i in range(len(poems_batch)):
        reply = tokenizer.decode(out[i][padded_len:], skip_special_tokens=True)
        theme = _normalize_theme(reply)
        themes.append(theme)
    return themes


def load_filtered_poems(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_themes(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_themes(themes: Dict[str, str], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(themes, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract poem themes via local Qwen3-8B"
    )
    parser.add_argument(
        "--filtered", type=str, default="./data/training/filtered_poems.json",
        help="Input filtered_poems.json path",
    )
    parser.add_argument(
        "--themes", type=str, default="./data/training/poem_themes.json",
        help="Output (and resume) poem_themes.json path",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of poems to process (0 = all). Use for testing.",
    )
    parser.add_argument(
        "--model_path", type=str, default="./ckpt/Qwen3-8B",
        help="Local Qwen3-8B model path (default: ./ckpt/Qwen3-8B)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=("cuda", "cpu"),
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for inference (default: 16). Increase if GPU memory allows.",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    print("Model loaded.")

    poems = load_filtered_poems(args.filtered)
    themes = load_existing_themes(args.themes)

    to_process = [p for p in poems if p["id"] not in themes]
    if args.limit > 0:
        to_process = to_process[: args.limit]

    total = len(to_process)
    if total == 0:
        print("No more poems to process.")
        return

    batch_size = max(1, args.batch_size)
    print(f"Processing {total} poems in batches of {batch_size} (already cached: {len(themes)}).")
    for start in tqdm(range(0, total, batch_size), desc="Batches"):
        batch = to_process[start : start + batch_size]
        batch_themes = extract_themes_batch(model, tokenizer, batch)
        for poem, theme in zip(batch, batch_themes):
            themes[poem["id"]] = theme if theme else poem["title"]
        if (start + len(batch)) % 1000 < batch_size:
            save_themes(themes, args.themes)

    save_themes(themes, args.themes)
    print(f"\nThemes saved to {args.themes} ({len(themes)} entries).")


if __name__ == "__main__":
    main()
