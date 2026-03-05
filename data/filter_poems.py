"""
Filter Complete Tang Poetry (全唐诗) to regulated verse only and save for theme extraction.

load from chinese-poetry, filter for 五言/七言 绝句/律诗, deduplicate.
Outputs filtered_poems.json (title and poem_text in traditional script) for use by
extract_themes.py and build_dataset.py
"""

import os
import json
import re
import argparse
from typing import List, Dict, Tuple, Optional
from collections import Counter

from .utils import download_chinese_poetry, load_tang_poems

POEM_TYPE_MAP = {
    (4, 5): "五言绝句",
    (4, 7): "七言绝句",
    (8, 5): "五言律诗",
    (8, 7): "七言律诗",
}
LINE_PATTERN = re.compile(r'([^，。；！？、：\s]+)([，。；！？、：])')
CJK_RE = re.compile(r'^[\u4e00-\u9fff]+$')


def extract_lines(paragraphs: List[str]) -> List[Tuple[str, str]]:
    full_text = "".join(p.strip() for p in paragraphs)
    return LINE_PATTERN.findall(full_text)


def classify_regulated_verse(lines: List[Tuple[str, str]]) -> Optional[str]:
    num_lines = len(lines)
    if num_lines not in (4, 8):
        return None
    char_counts = {len(text) for text, _ in lines}
    if len(char_counts) != 1:
        return None
    chars_per_line = char_counts.pop()
    if chars_per_line not in (5, 7):
        return None
    for i, (_, punct) in enumerate(lines):
        expected = "，" if (i % 2 == 0) else "。"
        if punct != expected:
            return None
    for text, _ in lines:
        if not CJK_RE.match(text):
            return None
    return POEM_TYPE_MAP.get((num_lines, chars_per_line))


def reconstruct_poem_text(lines: List[Tuple[str, str]]) -> str:
    return "".join(text + punct for text, punct in lines)


def filter_poems(raw_poems: List[Dict]) -> List[Dict]:
    """Apply filters (remove non-regulated verse, dedupe, title cleanup); return one record per poem."""
    stats = Counter()
    seen_texts = set()
    filtered = []

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

        filtered.append({
            "id": str(len(filtered)),
            "title": title,
            "poem_text": poem_text,
            "poem_type": poem_type,
        })
        stats[f"kept_{poem_type}"] += 1

    stats["kept_poems"] = len(filtered)
    return filtered, stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter 全唐诗 to regulated verse and save for theme extraction"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data/chinese-poetry",
        help="Path to chinese-poetry repo root (default: ./data/chinese-poetry)",
    )
    parser.add_argument(
        "--output", type=str, default="./data/training/filtered_poems.json",
        help="Output JSON path (default: ./data/training/filtered_poems.json)",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Auto-download chinese-poetry repo if not present",
    )
    args = parser.parse_args()

    if args.download:
        tang_dir = download_chinese_poetry(args.data_dir)
    else:
        tang_dir = os.path.join(args.data_dir, "全唐诗")
        if not os.path.isdir(tang_dir):
            print(f"Error: {tang_dir} not found.")
            print("Run with --download or place chinese-poetry at --data_dir.")
            return

    raw_poems = load_tang_poems(tang_dir)
    filtered, stats = filter_poems(raw_poems)

    if not filtered:
        print("No valid regulated verse poems found. Exiting.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("  Filter Statistics")
    print("=" * 50)
    print(f"  Total raw poems:            {stats['total']:>6}")
    print(f"  Skipped (missing fields):   {stats['skip_missing_fields']:>6}")
    print(f"  Skipped (no lines):         {stats['skip_no_lines']:>6}")
    print(f"  Skipped (not regulated):    {stats['skip_not_regulated']:>6}")
    print(f"  Skipped (duplicate):        {stats['skip_duplicate']:>6}")
    print("-" * 50)
    for poem_type in POEM_TYPE_MAP.values():
        print(f"  {poem_type}:                {stats.get(f'kept_{poem_type}', 0):>6}")
    print(f"  Kept poems:                 {stats['kept_poems']:>6}")
    print("=" * 50)
    print(f"\n{len(filtered)} poems → {args.output}")
    print("Filter complete.")


if __name__ == "__main__":
    main()
