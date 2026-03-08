"""Self-contained prosody checker for classical Chinese regulated verse.

Uses `ckpt/single_char_tokens.json`, which already covers both simplified and
traditional characters, to build character2tone and character2rhyme-group lookup
tables, and implements rhyme + meter checking for the four regulated-verse poem types.

Fixes two issues present in `pingshui_rhyme.PoemStructureChecker`:

1. Polyphonic line-1 endings no longer force 首句入韵.  A polyphonic character
   at the end of line 1 is valid in *either* reading (ping = 首句入韵,
   ze = 首句不入韵), so neither case is treated as a failure.

2. Characters absent from the dictionary ('unknown') are treated as flexible
   in both the strict pattern check and the 对/粘 fallback meter check.
"""
from __future__ import annotations

import json
import re
import os
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.utils import metrical_patterns

_DEFAULT_CHAR_TOKEN_PATH = os.path.join(_PROJECT_ROOT, "ckpt", "single_char_tokens.json")

# Maps (chars_per_line, num_lines) to poem type key used in metrical_patterns
_POEM_TYPE_FROM_DIMS = {
    (5, 4): "五言绝句",
    (7, 4): "七言绝句",
    (5, 8): "五言律诗",
    (7, 8): "七言律诗",
}

# Tones accepted at each pattern symbol:
#   P = strict ping position  → ping, polyphonic, or unknown (flexible)
#   Z = strict ze   position  → ze,   polyphonic, or unknown (flexible)
#   * = flexible position     → anything
_TONE_MATCH = {
    "P": frozenset({"ping", "polyphonic", "unknown"}),
    "Z": frozenset({"ze",  "polyphonic", "unknown"}),
    "*": frozenset({"ping", "ze", "polyphonic", "unknown"}),
}


def _split_lines(poem_text: str) -> list[str]:
    """Split poem on Chinese/ASCII punctuation and newlines; discard empty segments."""
    segments = re.split(r"[，。！？；：、,.!?;:\n]+", poem_text)
    segments = [re.sub(r"\s+", "", s) for s in segments]
    return [s for s in segments if s]


def _variant_matches(
    pattern_lines: list[str],
    poem_lines: list[str],
    checker: PoemStructureChecker,
) -> bool:
    """Return True if every character in every line satisfies the variant pattern."""
    if len(pattern_lines) != len(poem_lines):
        return False
    for pattern, line in zip(pattern_lines, poem_lines):
        if len(pattern) != len(line):
            return False
        for pat_char, char in zip(pattern, line):
            if checker.classify(char) not in _TONE_MATCH[pat_char]:
                return False
    return True


class PoemStructureChecker:
    """Ping-ze / rhyme checker for regulated verse."""

    def __init__(self, char_token_path = None) -> None:
        path = Path(char_token_path) if char_token_path is not None else Path(_DEFAULT_CHAR_TOKEN_PATH)
        if not path.is_file():
            raise FileNotFoundError(f"single_char_tokens.json not found at {path}")

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        char_list = data["single_char_tokens"]
        id_list = data["use_token_ids"]
        ping_ids = set(data["tone_index"]["ping"])
        ze_ids = set(data["tone_index"]["ze"])

        # Invert rhyme_index: token_id to set of rhyme group names
        id2rhyme = defaultdict(set)
        for group_name, ids in data["rhyme_index"].items():
            for tid in ids:
                id2rhyme[tid].add(group_name)

        ping_chars = set()
        ze_chars = set()
        rhyme_groups: dict[str, set[str]] = {}

        for ch, tid in zip(char_list, id_list):
            if tid in ping_ids:
                ping_chars.add(ch)
            if tid in ze_ids:
                ze_chars.add(ch)
            if tid in id2rhyme:
                rhyme_groups[ch] = id2rhyme[tid]

        self._ping = frozenset(ping_chars)
        self._ze = frozenset(ze_chars)
        self._rhyme_groups = {c: frozenset(gs) for c, gs in rhyme_groups.items()}

    def classify(self, char: str) -> str:
        """Return `'ping'`, `'ze'`, `'polyphonic'`, or `'unknown'`."""
        is_ping = char in self._ping
        is_ze = char in self._ze
        if is_ping and is_ze:
            return "polyphonic"
        if is_ping:
            return "ping"
        if is_ze:
            return "ze"
        return "unknown"

    def do_rhyme(self, char1: str, char2: str) -> bool:
        """True if char1 and char2 share a core Pingshui rhyme category.

        The "core" is the segment after 聲 in the group name, e.g. `"一東"`
        from `"上平聲一東"`.  This is the same algorithm used by
        `RhymeChecker.do_rhyme` in the `pingshui_rhyme` package.
        """
        gs1 = self._rhyme_groups.get(char1)
        gs2 = self._rhyme_groups.get(char2)
        if not gs1 or not gs2:
            return False
        cores1 = {g.split("聲")[-1] for g in gs1}
        cores2 = {g.split("聲")[-1] for g in gs2}
        return not cores1.isdisjoint(cores2)

    def check_rhyming(self, poem_text: str) -> tuple[bool, str]:
        """Check Pingshui rhyme rules for regulated verse.

        Jueju (4 lines)

        * Lines 2 and 4 must end with ping characters and must rhyme with each other.
        * Line 3 must end with a ze (or unknown) character.
        * Line 1:

          - Unambiguously ping → 首句入韵; must rhyme with line 2.
          - ze or unknown → 首句不入韵; no rhyme requirement.
          - polyphonic → either reading is acceptable; no failure is
            raised regardless of whether it happens to rhyme with line 2.

        Lushi (8 lines)

        * Even lines (2, 4, 6, 8) must end with ping characters and all rhyme
          with line 2.
        * Odd lines (3, 5, 7) must end with ze (or unknown) characters and must
          *not* rhyme with even lines.
        * Line 1 follows the same first-line rule as jueju.
        """
        lines = _split_lines(poem_text)
        if len(lines) < 4:
            return False, "Poem must have at least 4 lines."
        if len(lines) not in (4, 8):
            return True, "Non-standard line count, rhyme check skipped."

        chars_per_line = len(lines[0])
        if chars_per_line not in (5, 7):
            return False, f"Unexpected line length {chars_per_line}; expected 5 or 7."

        end_tones = [self.classify(line[-1]) for line in lines]
        is_jueju = len(lines) == 4

        def can_be_ping(tone: str) -> bool:
            return tone in ("ping", "polyphonic")

        # ze and unknown are both acceptable as "ze" endings (unknown = flexible).
        def can_be_ze(tone: str) -> bool:
            return tone in ("ze", "polyphonic", "unknown")

        # Enforce 首句入韵 when the first-line character is unambiguously ping
        enforce_first_rhyme = end_tones[0] == "ping"

        if is_jueju:
            char2 = lines[1][-1]
            char4 = lines[3][-1]
            if not can_be_ping(end_tones[1]):
                return False, "Line 2 must end with a ping character."
            if not can_be_ping(end_tones[3]):
                return False, "Line 4 must end with a ping character."
            if not self.do_rhyme(char2, char4):
                return False, "Lines 2 and 4 must rhyme."
            if not can_be_ze(end_tones[2]):
                return False, "Line 3 must end with a ze character."
            if enforce_first_rhyme and not self.do_rhyme(lines[0][-1], char2):
                return False, "Line 1 must rhyme with lines 2 and 4 (首句入韵)."

        else:  # lushi
            even_idx = list(range(1, len(lines), 2))  # 1, 3, 5, 7
            odd_idx  = list(range(2, len(lines), 2))  # 2, 4, 6

            for i in even_idx:
                if not can_be_ping(end_tones[i]):
                    return False, f"Line {i + 1} must end with a ping character."

            char2 = lines[1][-1]
            for i in even_idx[1:]:
                if not self.do_rhyme(char2, lines[i][-1]):
                    return False, f"Line {i + 1} must rhyme with line 2."

            for i in odd_idx:
                if not can_be_ze(end_tones[i]):
                    return False, f"Line {i + 1} must end with a ze character."
                if self.do_rhyme(char2, lines[i][-1]):
                    return False, f"Line {i + 1} must not rhyme with even lines."

            if enforce_first_rhyme and not self.do_rhyme(lines[0][-1], char2):
                return False, "Line 1 must rhyme with even lines (首句入韵)."

        poem_form = "jueju" if is_jueju else "lushi"
        return True, f"Rhyming check passed ({poem_form})."

    def check_meter(self, poem_text: str) -> tuple[bool, str]:
        """Check the ping-ze tonal meter for regulated verse.

        First tries strict pattern matching against all 4 variants defined in
        `model.utils.metrical_patterns` (平起入韵, 平起不入韵, 仄起入韵,
        仄起不入韵).  At every position:

        * `P`: character must be ping, polyphonic, or unknown.
        * `Z`: character must be ze, polyphonic, or unknown.
        * `*`: any character is accepted.

        If no variant matches, falls back to the 对/粘 alternation check on
        positions 2 and 4 (and 6 for seven-character poems).  Characters
        classified as `'polyphonic'` or `'unknown'` are skipped in the
        fallback check.
        """
        lines = _split_lines(poem_text)
        if len(lines) < 4:
            return False, "Poem must have at least 4 lines."
        if len(lines) not in (4, 8):
            return True, "Non-standard line count, meter check skipped."

        chars_per_line = len(lines[0])
        if chars_per_line not in (5, 7):
            return False, f"Unexpected line length {chars_per_line}; expected 5 or 7."

        poem_type_key = _POEM_TYPE_FROM_DIMS.get((chars_per_line, len(lines)))
        if poem_type_key is None or poem_type_key not in metrical_patterns:
            return True, "No pattern defined for this poem type, meter check skipped."

        # Strict pattern matching: try all 4 variants
        for variant in metrical_patterns[poem_type_key]:
            if _variant_matches(variant["lines"], lines, self):
                return True, f"Meter matches variant '{variant['name']}'."

        # Fallback: 对/粘 alternation check.
        # 对 (even-indexed adjacent pairs 0-1, 2-3 ...): tones must differ.
        # 粘 (odd-indexed adjacent pairs 1-2, 3-4 ...): tones must agree.
        # Skip any position where either character is polyphonic or unknown.
        key_pos = [1, 3] if chars_per_line == 5 else [1, 3, 5]
        for i in range(len(lines) - 1):
            line_a, line_b = lines[i], lines[i + 1]
            for pos in key_pos:
                ta = self.classify(line_a[pos])
                tb = self.classify(line_b[pos])
                if ta in ("polyphonic", "unknown") or tb in ("polyphonic", "unknown"):
                    continue
                if i % 2 == 0:  # 对: tones must be opposite
                    if ta == tb:
                        return False, (
                            f"对 violation: lines {i + 1}/{i + 2} both '{ta}' "
                            f"at position {pos + 1}."
                        )
                else:  # 粘: tones must be the same
                    if ta != tb:
                        return False, (
                            f"粘 violation: lines {i + 1}/{i + 2} differ at "
                            f"position {pos + 1} ('{ta}' vs '{tb}')."
                        )

        return True, "Meter check passed (对/粘 fallback)."

