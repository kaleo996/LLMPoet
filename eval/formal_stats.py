"""
Step 2 of CharPoet-style evaluation: read eval_poems.json, compute format
accuracy and Pingshui rhyme/meter stats using PoemStructureChecker.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import opencc
import time
s2t_converter = opencc.OpenCC("s2t")

# #region agent log
_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug-fda10f.log")
def _debug_log(location, message, data, hypothesis=""):
    import json as _json
    entry = {"sessionId": "fda10f", "location": location, "message": message, "data": data, "hypothesisId": hypothesis, "timestamp": int(time.time() * 1000)}
    with open(_LOG_PATH, "a", encoding="utf-8") as _f:
        _f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
# #endregion

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.utils import masked_poem_dict
from pingshui_rhyme import PoemStructureChecker


def parse_template(masked_poem: str):
    """Return segments (char_count, punct) and expected line lengths (chars per line)."""
    segments = []
    placeholder = "<|extra_1|>"
    pattern = r"(<\|extra_1\|>)+([，。、；]?)"
    for match in re.finditer(pattern, masked_poem):
        punct = match.group(2) or ""
        placeholder_len = len(match.group(0)) - len(punct)
        char_count = placeholder_len // len(placeholder)
        segments.append((char_count, punct))
    expected_lengths = [seg[0] for seg in segments]
    return segments, expected_lengths


def get_expected_line_lengths(poem_type: str) -> list[int]:
    """Return list of expected character counts per line for the given poem type."""
    if poem_type not in masked_poem_dict:
        return []
    _, lengths = parse_template(masked_poem_dict[poem_type])
    return lengths


def build_poem_text_for_checker(poem_text: str) -> str:
    """Convert generated poem string to format expected by PoemStructureChecker.

    Converts simplified Chinese to traditional Chinese because PoemStructureChecker's
    rhyme/tone dictionary uses traditional characters exclusively.
    """
    return s2t_converter.convert(poem_text)


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

        fmt_ok = check_format_accuracy(poem_text, poem_type)
        if fmt_ok:
            stats["format_ok"] += 1

        checker_text = build_poem_text_for_checker(poem_text)
        if not checker_text.strip():
            stats["skip_no_paragraphs"] += 1
            # #region agent log
            _debug_log("formal_stats.py:skip", "Skipped empty poem", {"poem_type": poem_type, "theme": rec.get("theme", "")}, hypothesis="C")
            # #endregion
            continue

        rhyming_reason = ""
        try:
            rhyming_ok, rhyming_reason = checker.check_poem_rhyming(checker_text)
        except Exception as e:
            stats["error_rhyming"] += 1
            rhyming_ok = False
            rhyming_reason = f"exception: {e}"
        if rhyming_ok:
            stats["rhyming_ok"] += 1

        meter_reason = ""
        try:
            meter_ok, meter_reason = checker.check_poem_pingze_meter(checker_text)
        except Exception as e:
            stats["error_meter"] += 1
            meter_ok = False
            meter_reason = f"exception: {e}"
        if meter_ok:
            stats["meter_ok"] += 1

        if rhyming_ok and meter_ok:
            stats["both_ok"] += 1

        # #region agent log
        lines = checker.clean_poem(checker_text)
        line_endings_info = []
        for li, line in enumerate(lines):
            if line:
                last_char = line[-1]
                tone = checker.classifier.classify(last_char)
                rhyme_groups = checker.rhyme_checker.get_rhyme_group(last_char)
                line_endings_info.append({
                    "line_idx": li,
                    "last_char": last_char,
                    "tone": tone,
                    "rhyme_groups": [(g[0], g[1], g[2]) for g in rhyme_groups] if rhyme_groups else None,
                    "line_text": line,
                })
        _debug_log("formal_stats.py:check", "Poem check result", {
            "poem_type": poem_type,
            "theme": rec.get("theme", ""),
            "poem_text_sc": poem_text,
            "checker_text_tc": checker_text,
            "format_ok": fmt_ok,
            "rhyming_ok": rhyming_ok,
            "rhyming_reason": rhyming_reason,
            "meter_ok": meter_ok,
            "meter_reason": meter_reason,
            "line_endings": line_endings_info,
        }, hypothesis="A,B,C,D,E")
        # #endregion

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

    if not os.path.isfile(args.input):
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
    by_type = defaultdict(list)
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
