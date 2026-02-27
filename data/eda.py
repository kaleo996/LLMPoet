"""
Use pingshui_rhyme.PoemStructureChecker to count how many poems in
Complete Tang Poetry (全唐诗) strictly follow regulated verse rules.
"""

import os
import argparse
from collections import Counter

from pingshui_rhyme import PoemStructureChecker

from .utils import download_chinese_poetry, load_tang_poems


def build_poem_text(paragraphs):
    """Join a poem's paragraph list into a single string for checking.

    We simply join all non-empty paragraph strings with newlines and add
    a trailing newline, without enforcing any particular external shape.
    """
    lines = [p.strip() for p in paragraphs if p and p.strip()]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def analyze_poems(raw_poems, limit=None):
    """Run PoemStructureChecker over poems and collect statistics.

    Args:
        raw_poems: List of poem dicts loaded from 全唐诗 JSON files.
        limit: Optional int, max number of poems to process (for debugging).

    Returns:
        stats: Counter with keys:
            - "total": total poems processed
            - "skip_no_paragraphs": missing or empty paragraphs
            - "rhyming_ok": poems passing check_poem_rhyming
            - "meter_ok": poems passing check_poem_pingze_meter
            - "both_ok": poems passing both checks
            - "error_rhyming": exceptions raised during rhyming check
            - "error_meter": exceptions raised during ping-ze check
    """
    checker = PoemStructureChecker()
    stats = Counter()

    for idx, poem in enumerate(raw_poems):
        if limit is not None and idx >= limit:
            break

        stats["total"] += 1
        paragraphs = poem.get("paragraphs", [])
        if not paragraphs:
            stats["skip_no_paragraphs"] += 1
            continue

        poem_text = build_poem_text(paragraphs)
        if not poem_text.strip():
            stats["skip_no_paragraphs"] += 1
            continue

        # Rhyming check (Jueju / Lushi rhyme scheme)
        try:
            rhyming_ok, _ = checker.check_poem_rhyming(poem_text)
        except Exception:
            # Some poems may trigger internal errors in the library; count and skip.
            stats["error_rhyming"] += 1
            rhyming_ok = False
        if rhyming_ok:
            stats["rhyming_ok"] += 1

        # Ping-ze meter check
        try:
            meter_ok, _ = checker.check_poem_pingze_meter(poem_text)
        except Exception:
            stats["error_meter"] += 1
            meter_ok = False
        if meter_ok:
            stats["meter_ok"] += 1

        if rhyming_ok and meter_ok:
            stats["both_ok"] += 1

        if (idx + 1) % 10000 == 0:
            print(
                f"[progress] Processed {idx + 1} poems - "
                f"rhyming_ok={stats['rhyming_ok']}, "
                f"meter_ok={stats['meter_ok']}, "
                f"both_ok={stats['both_ok']}"
            )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Count how many poems in Complete Tang Poetry strictly follow "
            "Pingshui regulated verse rules using PoemStructureChecker."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/chinese-poetry",
        help=(
            "Path to chinese-poetry repo root (default: ./data/chinese-poetry). "
            "The 全唐诗 JSON files are expected under a 子目录 named 全唐诗."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Auto-download the chinese-poetry repo via git sparse checkout "
            "if not already present at --data_dir."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of poems to process (for quick testing). "
            "Default: process all poems."
        ),
    )
    args = parser.parse_args()

    # Locate or download data (reuse logic from prepare_dataset.py)
    if args.download:
        tang_dir = download_chinese_poetry(args.data_dir)
    else:
        tang_dir = os.path.join(args.data_dir, "全唐诗")
        if not os.path.isdir(tang_dir):
            print(f"Error: {tang_dir} not found.")
            print(f"Either place the chinese-poetry repo at {args.data_dir},")
            print(f"or run with --download to auto-download.")
            return

    print(f"[load] Loading poems from: {tang_dir}")
    raw_poems = load_tang_poems(tang_dir)
    if not raw_poems:
        print("No poems loaded from 全唐诗. Exiting.")
        return

    if args.limit is not None:
        print(f"[info] Processing at most {args.limit} poems (out of {len(raw_poems)})")

    stats = analyze_poems(raw_poems, limit=args.limit)

    print("\n" + "=" * 50)
    print("  Pingshui Regulated Verse Statistics")
    print("=" * 50)
    print(f"  Total poems processed:        {stats['total']:>8}")
    print(f"  Skipped (no paragraphs):      {stats['skip_no_paragraphs']:>8}")
    print("-" * 50)
    print(f"  Rhyming check passed:         {stats['rhyming_ok']:>8}")
    print(f"  Ping-ze meter check passed:   {stats['meter_ok']:>8}")
    print(f"  Both checks passed (严格格律):  {stats['both_ok']:>8}")
    print("-" * 50)
    print(f"  Rhyming check errors:         {stats['error_rhyming']:>8}")
    print(f"  Ping-ze check errors:         {stats['error_meter']:>8}")
    print("=" * 50)


if __name__ == "__main__":
    main()

