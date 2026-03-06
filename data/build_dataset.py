"""
Build final train/eval JSONL from filtered poems and LLM-extracted themes.

Reads filtered_poems.json and poem_themes.json, uses theme (or title fallback)
in the prompt template for both traditional and simplified (OpenCC) samples,
then splits and saves train.jsonl and eval.jsonl.
"""

import os
import json
import random
import argparse
from typing import List, Dict
from collections import Counter

import opencc

from model.utils import (
    masked_poem_dict,
    poetry_prompt_template_sc,
    poetry_prompt_template_tc,
    get_poem_type_display,
)


def load_filtered_poems(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_themes(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_sample(poem_type: str, title: str, poem_text: str,
                   script: str = "traditional") -> str:
    """Build a training text using the prompt template.

    Args:
        poem_type: Internal key (simplified), e.g. "五言绝句".
        title: Poem title in the target script.
        poem_text: Poem body in the target script.
        script: "simplified" or "traditional".
    """
    masked_poem = masked_poem_dict[poem_type]
    template = poetry_prompt_template_tc if script == "traditional" else poetry_prompt_template_sc
    display_type = get_poem_type_display(poem_type, script)

    return template.format_map({
        "user_prompt": title,
        "masked_poem": masked_poem,
        "poem_type": display_type,
    }) + poem_text + "<|im_end|>\n"


def build_samples(poems: List[Dict], themes: Dict[str, str]) -> List[Dict]:
    """Build tc + sc samples using theme in prompt; fallback to title if theme missing."""
    t2s = opencc.OpenCC("t2s")
    samples = []

    for poem in poems:
        poem_type = poem["poem_type"]
        title = poem["title"]
        poem_text = poem["poem_text"]
        theme = themes.get(poem["id"])
        if not theme:
            continue
        # theme = themes.get(poem["id"], title)

        # Traditional
        tc_text = format_sample(poem_type, theme, poem_text, script="traditional")
        samples.append({
            "text": tc_text,
            "poem_type": poem_type,
            "script": "traditional",
        })

        # Simplified (OpenCC)
        sc_theme = t2s.convert(theme)
        sc_poem_text = t2s.convert(poem_text)
        sc_text = format_sample(poem_type, sc_theme, sc_poem_text, script="simplified")
        samples.append({
            "text": sc_text,
            "poem_type": poem_type,
            "script": "simplified",
        })

    return samples


def split_and_save(
    samples: List[Dict],
    output_dir: str,
    eval_ratio: float,
    seed: int,
):
    """Shuffle, split into train/eval, and write JSONL files."""
    random.seed(seed)
    random.shuffle(samples)

    n_eval = max(1, int(len(samples) * eval_ratio))
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    eval_path = os.path.join(output_dir, "eval.jsonl")

    for path, data in [(train_path, train_samples), (eval_path, eval_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[save] Training samples:   {len(train_samples):>6}  →  {train_path}")
    print(f"[save] Evaluation samples: {len(eval_samples):>6}  →  {eval_path}")

    # Training set distribution
    type_dist = Counter(s["poem_type"] for s in train_samples)
    script_dist = Counter(s["script"] for s in train_samples)
    print("\n[save] Training set distribution:")
    print("  By type:")
    for pt, count in type_dist.most_common():
        print(f"    {pt}: {count}")
    print("  By script:")
    for sc, count in script_dist.most_common():
        print(f"    {sc}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Build train/eval JSONL from filtered poems and themes"
    )
    parser.add_argument(
        "--filtered", type=str, default="./data/training/filtered_poems.json",
        help="Input filtered_poems.json path",
    )
    parser.add_argument(
        "--themes", type=str, default="./data/training/poem_themes.json",
        help="Input poem_themes.json path",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/training",
        help="Output directory for train.jsonl and eval.jsonl (default: ./data/training)",
    )
    parser.add_argument(
        "--eval_ratio", type=float, default=0.05,
        help="Fraction for evaluation (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split (default: 42)",
    )
    parser.add_argument(
        "--preview", type=int, default=3,
        help="Number of sample previews (0 to disable)",
    )
    args = parser.parse_args()

    poems = load_filtered_poems(args.filtered)
    themes = load_themes(args.themes)

    missing = sum(1 for p in poems if p["id"] not in themes)
    if missing > 0:
        print(f"Note: {missing} poems have no theme (using title as fallback).")

    samples = build_samples(poems, themes)
    print(f"Built {len(samples)} samples ({len(poems)} poems × 2 scripts).")

    if args.preview > 0:
        previews = random.sample(samples, min(args.preview, len(samples)))
        print(f"\n--- Preview ({args.preview} samples) ---")
        for i, s in enumerate(previews, 1):
            tag = "繁" if s["script"] == "traditional" else "简"
            print(f"\n[{i}] [{s['poem_type']}][{tag}]")
            print(s["text"])

    split_and_save(samples, args.output_dir, args.eval_ratio, args.seed)
    print("\nDataset build complete.")


if __name__ == "__main__":
    main()
