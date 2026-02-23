"""
Prepare training dataset from Complete Tang Poetry (全唐诗) for fine-tuning.

Downloads (or loads) the chinese-poetry dataset, filters for regulated verse
(格律诗: 五言/七言 绝句/律诗), formats each poem into the project's chat
template, and outputs train/eval JSONL splits.
"""

import os
import json
import re
import glob
import argparse
import random
import subprocess
from typing import List, Tuple, Optional, Dict
from collections import Counter, defaultdict

import opencc

from utils import (
    masked_poem_dict,
    poetry_prompt_template_sc,
    poetry_prompt_template_tc,
    get_poem_type_display,
)

POEM_TYPE_MAP = {
    (4, 5): "五言绝句",
    (4, 7): "七言绝句",
    (8, 5): "五言律诗",
    (8, 7): "七言律诗",
}

# Regex: a string of non-punctuation characters followed by exactly one punctuation mark
LINE_PATTERN = re.compile(r'([^，。；！？、：\s]+)([，。；！？、：])')

# Characters we consider valid in a poem line (CJK Unified Ideographs)
CJK_RE = re.compile(r'^[\u4e00-\u9fff]+$')


# Data download
def download_chinese_poetry(target_dir: str) -> str:
    """Download chinese-poetry repo via git sparse checkout (全唐诗 only).

    Returns:
        Path to the 全唐诗 directory.
    """
    tang_dir = os.path.join(target_dir, "全唐诗")
    if os.path.isdir(tang_dir) and any(f.endswith(".json") for f in os.listdir(tang_dir)):
        print(f"[download] 全唐诗 already exists at {tang_dir}, skipping download.")
        return tang_dir

    print(f"[download] Cloning chinese-poetry repo (sparse checkout) into {target_dir} ...")
    os.makedirs(os.path.dirname(target_dir) or ".", exist_ok=True)

    # Sparse clone: only metadata, no blobs yet
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "--filter=blob:none", "--sparse",
            "https://github.com/chinese-poetry/chinese-poetry.git",
            target_dir,
        ],
        check=True,
    )
    # Check out only the 全唐诗 folder
    subprocess.run(
        ["git", "sparse-checkout", "set", "全唐诗"],
        cwd=target_dir,
        check=True,
    )
    print(f"[download] Done. Tang poetry files at {tang_dir}")
    return tang_dir


def extract_lines(paragraphs: List[str]) -> List[Tuple[str, str]]:
    """Split poem paragraphs into individual (text, punctuation) tuples.

    Example:
        ["白日依山尽，黄河入海流。", "欲穷千里目，更上一层楼。"]
        → [("白日依山尽", "，"), ("黄河入海流", "。"),
           ("欲穷千里目", "，"), ("更上一层楼", "。")]
    """
    full_text = "".join(p.strip() for p in paragraphs)
    return LINE_PATTERN.findall(full_text)


def classify_regulated_verse(lines: List[Tuple[str, str]]) -> Optional[str]:
    """Return poem type string if `lines` form a valid regulated verse, else None.

    Checks:
        1. Line count is 4 (绝句) or 8 (律诗).
        2. All lines have the same character count (5 or 7).
        3. Punctuation alternates `，` and `。` correctly.
        4. All characters are CJK ideographs (no digits, Latin, annotations).
    """
    num_lines = len(lines)
    if num_lines not in (4, 8):
        return None

    # All lines must have the same character count (5 or 7)
    char_counts = {len(text) for text, _ in lines}
    if len(char_counts) != 1:
        return None
    chars_per_line = char_counts.pop()
    if chars_per_line not in (5, 7):
        return None

    # Punctuation must alternate `，` and `。`
    for i, (_, punct) in enumerate(lines):
        expected = "，" if (i % 2 == 0) else "。"
        if punct != expected:
            return None

    # Every character must be a CJK ideograph
    for text, _ in lines:
        if not CJK_RE.match(text):
            return None

    return POEM_TYPE_MAP.get((num_lines, chars_per_line))


def reconstruct_poem_text(lines: List[Tuple[str, str]]) -> str:
    """Join lines back into a single poem string with punctuation."""
    return "".join(text + punct for text, punct in lines)


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


def load_tang_poems(tang_dir: str) -> List[Dict]:
    """Load all poem objects from 全唐诗 JSON files."""
    json_files = sorted(glob.glob(os.path.join(tang_dir, "poet.tang.*.json")))
    if not json_files:
        # Fallback: try any JSON file in the directory
        json_files = sorted(glob.glob(os.path.join(tang_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {tang_dir}")

    poems = []
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                poems.extend(data)
            elif isinstance(data, dict):
                poems.append(data)
    print(f"[load] Loaded {len(poems)} raw poems from {len(json_files)} files.")
    return poems


def process_poems(raw_poems: List[Dict]) -> List[Dict]:
    """Filter, classify, deduplicate, and format poems into training samples."""
    t2s = opencc.OpenCC("t2s")
    stats = Counter()
    seen_texts = set()
    samples_by_type = defaultdict(list)

    for poem in raw_poems:
        stats["total"] += 1
        title = poem.get("title", "").strip()
        paragraphs = poem.get("paragraphs", [])

        if not title or not paragraphs:
            stats["skip_missing_fields"] += 1
            continue

        lines = extract_lines(paragraphs)
        if not lines:
            stats["skip_no_lines"] += 1
            continue

        poem_type = classify_regulated_verse(lines)
        if poem_type is None:
            stats["skip_not_regulated"] += 1
            continue

        poem_text = reconstruct_poem_text(lines)
        if poem_text in seen_texts:
            stats["skip_duplicate"] += 1
            continue
        seen_texts.add(poem_text)

        title = re.sub(r'[\(（].*?[\)）]', '', title).strip()
        if not title:
            stats["skip_empty_title"] += 1
            continue

        # Traditional sample (original data from 全唐诗)
        tc_text = format_sample(poem_type, title, poem_text, script="traditional")
        samples_by_type[poem_type].append({
            "text": tc_text,
            "poem_type": poem_type,
            "script": "traditional",
        })

        # Simplified sample (converted via opencc)
        sc_title = t2s.convert(title)
        sc_poem_text = t2s.convert(poem_text)
        sc_text = format_sample(poem_type, sc_title, sc_poem_text, script="simplified")
        samples_by_type[poem_type].append({
            "text": sc_text,
            "poem_type": poem_type,
            "script": "simplified",
        })

        stats[f"kept_{poem_type}"] += 1

    stats["kept_poems"] = sum(len(v) // 2 for v in samples_by_type.values())
    stats["kept_samples"] = sum(len(v) for v in samples_by_type.values())

    print("\n" + "=" * 55)
    print("  Processing Statistics")
    print("=" * 55)
    print(f"  Total raw poems:            {stats['total']:>6}")
    print(f"  Skipped (missing fields):   {stats['skip_missing_fields']:>6}")
    print(f"  Skipped (no lines):         {stats['skip_no_lines']:>6}")
    print(f"  Skipped (not regulated):    {stats['skip_not_regulated']:>6}")
    print(f"  Skipped (duplicate):        {stats['skip_duplicate']:>6}")
    print(f"  Skipped (empty title):      {stats['skip_empty_title']:>6}")
    print("-" * 55)
    for poem_type in POEM_TYPE_MAP.values():
        count = len(samples_by_type.get(poem_type, []))
        print(f"  {poem_type} (tc+sc):       {count:>6}")
    print(f"  Kept poems (unique):        {stats['kept_poems']:>6}")
    print(f"  Kept samples (tc+sc):       {stats['kept_samples']:>6}")
    print("=" * 55)

    all_samples = []
    for poem_type in POEM_TYPE_MAP.values():
        all_samples.extend(samples_by_type.get(poem_type, []))
    return all_samples


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


def preview_samples(samples: List[Dict], n: int = 3):
    """Print a few formatted samples for visual verification."""
    print(f"\n{'=' * 50}")
    print(f"  Preview ({n} random samples)")
    print(f"{'=' * 50}")
    previews = random.sample(samples, min(n, len(samples)))
    for i, sample in enumerate(previews, 1):
        script_tag = "繁" if sample.get("script") == "traditional" else "简"
        print(f"\n--- Sample {i} [{sample['poem_type']}][{script_tag}] ---")
        print(sample["text"])
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 全唐诗 regulated verse dataset for LoRA fine-tuning"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data/chinese-poetry",
        help="Path to chinese-poetry repo root, or download target (default: ./data/chinese-poetry)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/training",
        help="Output directory for train.jsonl and eval.jsonl (default: ./data/training)",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Auto-download the chinese-poetry repo via git if not present",
    )
    parser.add_argument(
        "--eval_ratio", type=float, default=0.05,
        help="Fraction of data reserved for evaluation (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splitting (default: 42)",
    )
    parser.add_argument(
        "--preview", type=int, default=3,
        help="Number of sample previews to display (default: 3, 0 to disable)",
    )
    args = parser.parse_args()

    # Locate or download data
    if args.download:
        tang_dir = download_chinese_poetry(args.data_dir)
    else:
        tang_dir = os.path.join(args.data_dir, "全唐诗")
        if not os.path.isdir(tang_dir):
            print(f"Error: {tang_dir} not found.")
            print(f"Either place the chinese-poetry repo at {args.data_dir},")
            print(f"or run with --download to auto-download.")
            return

    # Load raw poems, filter, classify, format
    raw_poems = load_tang_poems(tang_dir)
    samples = process_poems(raw_poems)
    if not samples:
        print("No valid regulated verse poems found. Exiting.")
        return

    if args.preview > 0:
        preview_samples(samples, n=args.preview)

    split_and_save(samples, args.output_dir, args.eval_ratio, args.seed)

    print("\n[done] Dataset preparation complete!")


if __name__ == "__main__":
    main()
