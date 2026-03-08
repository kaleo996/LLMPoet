"""
Step 2 of CharPoet-style evaluation: read eval_poems.json, compute format
accuracy and Pingshui rhyme/meter stats using local `PoemStructureChecker`.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from eval.poem_structure_checker import PoemStructureChecker
from model.utils import get_poem_structure


def get_expected_line_lengths(poem_type: str) -> list[int]:
    """Return list of expected character counts per line for the given poem type."""
    try:
        num_lines, chars_per_line = get_poem_structure(poem_type)
        return [chars_per_line] * num_lines
    except ValueError:
        return []


def count_chars_per_line(poem_text: str) -> list[int]:
    """Split by Chinese punctuation，。、；and count characters per segment.

    Poems may be on a single line with only punctuation as separators (no newlines),
    so we split on punctuation rather than newlines.
    """
    segments = re.split(r'[，。、；\n]+', poem_text.strip())
    segments = [s.strip() for s in segments if s.strip()]
    return [len(s) for s in segments]


def check_format_accuracy(poem_text: str, poem_type: str) -> bool:
    """True if every line has the expected number of characters."""
    expected = get_expected_line_lengths(poem_type)
    if not expected:
        return False
    actual = count_chars_per_line(poem_text)
    return actual == expected


def load_poems(path: str) -> tuple[list, list]:
    """Load idiom and instruction lists from eval_poems.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idiom = data.get("idiom", [])
    instruction = data.get("instruction", [])
    return idiom, instruction


def run_stats(poem_list: list, checker: PoemStructureChecker) -> dict:
    """Run format + rhyme + meter checks on a list of poem records. Return stats dict."""
    stats = Counter()
    for rec in poem_list:
        poem_text = rec.get("poem_text", "")
        poem_type = rec.get("poem_type", "")

        stats["total"] += 1

        if check_format_accuracy(poem_text, poem_type):
            stats["format_ok"] += 1

        if not poem_text.strip():
            stats["skip_no_paragraphs"] += 1
            continue

        try:
            rhyming_ok, _ = checker.check_rhyming(poem_text)
        except Exception:
            stats["error_rhyming"] += 1
            rhyming_ok = False
        if rhyming_ok:
            stats["rhyming_ok"] += 1

        try:
            meter_ok, _ = checker.check_meter(poem_text)
        except Exception:
            stats["error_meter"] += 1
            meter_ok = False
        if meter_ok:
            stats["meter_ok"] += 1

        if rhyming_ok and meter_ok:
            stats["both_ok"] += 1

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description="Compute format accuracy and Pingshui rhyme/meter stats on eval_poems.json."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "eval", "eval_poems.json"),
        help="Path to eval_poems.json from stage 1.",
    )
    args = parser.parse_args()

    if not Path(args.input).is_file():
        print(f"Error: {args.input} not found. Run eval/generate_for_eval.py first.")
        sys.exit(1)

    idiom_list, instruction_list = load_poems(args.input)
    checker = PoemStructureChecker()

    all_records = []
    if idiom_list:
        all_records.extend(idiom_list)
    if instruction_list:
        all_records.extend(instruction_list)

    if not all_records:
        print("No poems in input. Exiting.")
        return

    # Overall
    overall = run_stats(all_records, checker)
    # By theme_type
    idiom_stats = run_stats(idiom_list, checker) if idiom_list else {}
    instr_stats = run_stats(instruction_list, checker) if instruction_list else {}

    # By poem_type
    by_type: dict[str, list] = defaultdict(list)
    for rec in all_records:
        by_type[rec.get("poem_type", "")].append(rec)
    type_stats = {pt: run_stats(by_type[pt], checker) for pt in sorted(by_type) if by_type[pt]}

    print("\n" + "=" * 50)
    print("  Format & Pingshui Regulated Verse Statistics")
    print("=" * 50)
    print(f"  Total poems:                  {overall.get('total', 0):>8}")
    print(f"  Format accuracy (correct):    {overall.get('format_ok', 0):>8}")
    if overall.get("total"):
        fmt_rate = overall.get("format_ok", 0) / overall["total"]
        print(f"  Format accuracy rate:         {fmt_rate:.4f}")
    print("-" * 50)
    print(f"  Rhyming check passed:         {overall.get('rhyming_ok', 0):>8}")
    print(f"  Ping-ze meter passed:         {overall.get('meter_ok', 0):>8}")
    print(f"  Both passed (严格格律):        {overall.get('both_ok', 0):>8}")
    print("-" * 50)
    print(f"  Skipped (no paragraphs):      {overall.get('skip_no_paragraphs', 0):>8}")
    print(f"  Rhyming check errors:         {overall.get('error_rhyming', 0):>8}")
    print(f"  Ping-ze check errors:         {overall.get('error_meter', 0):>8}")
    print("=" * 50)

    if idiom_stats:
        print("\n  By theme_type=idiom (keyword):")
        print(f"    total={idiom_stats.get('total', 0)} format_ok={idiom_stats.get('format_ok', 0)} "
              f"rhyming_ok={idiom_stats.get('rhyming_ok', 0)} meter_ok={idiom_stats.get('meter_ok', 0)} both_ok={idiom_stats.get('both_ok', 0)}")
    if instr_stats:
        print("  By theme_type=instruction:")
        print(f"    total={instr_stats.get('total', 0)} format_ok={instr_stats.get('format_ok', 0)} "
              f"rhyming_ok={instr_stats.get('rhyming_ok', 0)} meter_ok={instr_stats.get('meter_ok', 0)} both_ok={instr_stats.get('both_ok', 0)}")

    if type_stats:
        print("\n  By poem_type:")
        for pt, st in type_stats.items():
            t = st.get("total", 0)
            fo = st.get("format_ok", 0)
            ro = st.get("rhyming_ok", 0)
            mo = st.get("meter_ok", 0)
            bo = st.get("both_ok", 0)
            print(f"    {pt}: total={t} format_ok={fo} rhyming_ok={ro} meter_ok={mo} both_ok={bo}")


if __name__ == "__main__":
    main()
