import os
import re
from typing import Optional

try:
    import opencc
except ImportError:
    opencc = None
try:
    from pingshui_rhyme import PoemStructureChecker
except ImportError:
    PoemStructureChecker = None

from model.utils import masked_poem_dict

from .schemas import EvaluationReport


if opencc is not None:
    _S2T = opencc.OpenCC("s2t")
else:
    _S2T = None
_CHECKER: Optional[PoemStructureChecker] = None


def get_checker() -> PoemStructureChecker:
    if PoemStructureChecker is None:
        raise ImportError("pingshui_rhyme is required for rhyme and meter checking")
    global _CHECKER
    if _CHECKER is None:
        _CHECKER = PoemStructureChecker()
    return _CHECKER


def parse_template(masked_poem: str) -> tuple[list[tuple[int, str]], list[int]]:
    segments = []
    placeholder = "<|extra_1|>"
    pattern = r"(<\|extra_1\|>)+([，。、；]?)"
    for match in re.finditer(pattern, masked_poem):
        punct = match.group(2) or ""
        placeholder_len = len(match.group(0)) - len(punct)
        char_count = placeholder_len // len(placeholder)
        segments.append((char_count, punct))
    return segments, [seg[0] for seg in segments]


def expected_line_lengths(poem_type: str) -> list[int]:
    if poem_type not in masked_poem_dict:
        return []
    _, lengths = parse_template(masked_poem_dict[poem_type])
    return lengths


def count_chars_per_line(poem_text: str) -> list[int]:
    segments = re.split(r"[，。、；\n]+", poem_text.strip())
    segments = [seg.strip() for seg in segments if seg.strip()]
    return [len(seg) for seg in segments]


def normalize_poem_text(poem_text: str) -> str:
    text = poem_text.strip()
    text = text.replace("\r", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        return lines[0]
    return "".join(lines)


def _quality_score(poem_text: str, user_prompt: str) -> float:
    score = 0.0
    text = normalize_poem_text(poem_text)
    if text:
        score += min(len(set(text)) / max(len(text), 1), 1.0) * 5
    prompt_chars = {ch for ch in user_prompt if "\u4e00" <= ch <= "\u9fff"}
    if prompt_chars:
        overlap = sum(1 for ch in text if ch in prompt_chars)
        score += min(overlap, len(prompt_chars)) * 0.5
    repeated_bigrams = 0
    for idx in range(len(text) - 1):
        if text[idx: idx + 2] == text[max(0, idx - 2): max(0, idx)]:
            repeated_bigrams += 1
    score -= repeated_bigrams * 0.5
    return round(score, 4)


def evaluate_poem(poem_text: str, poem_type: str, user_prompt: str = "") -> EvaluationReport:
    poem_text = normalize_poem_text(poem_text)
    expected_lengths = expected_line_lengths(poem_type)
    actual_lengths = count_chars_per_line(poem_text)

    report = EvaluationReport(
        line_lengths=actual_lengths,
        expected_line_lengths=expected_lengths,
    )
    report.line_length_ok = actual_lengths == expected_lengths
    report.format_ok = report.line_length_ok
    if not report.line_length_ok:
        report.failure_reasons.append(
            f"line_length_mismatch: expected={expected_lengths}, actual={actual_lengths}"
        )

    checker_text = _S2T.convert(poem_text) if _S2T is not None else poem_text

    if not checker_text.strip():
        report.failure_reasons.append("empty_poem")
    elif PoemStructureChecker is None:
        report.detail["rhyme_meter_warning"] = "pingshui_rhyme not installed; skipped rhyme and meter checks"
    else:
        checker = get_checker()
        try:
            rhyming_ok, rhyming_reason = checker.check_poem_rhyming(checker_text)
            report.rhyming_ok = bool(rhyming_ok)
            report.detail["rhyming_reason"] = rhyming_reason
            if not report.rhyming_ok:
                report.failure_reasons.append(f"rhyming_failed: {rhyming_reason}")
        except Exception as exc:
            report.detail["rhyming_reason"] = str(exc)
            report.failure_reasons.append(f"rhyming_error: {exc}")

        try:
            meter_ok, meter_reason = checker.check_poem_pingze_meter(checker_text)
            report.meter_ok = bool(meter_ok)
            report.detail["meter_reason"] = meter_reason
            if not report.meter_ok:
                report.failure_reasons.append(f"meter_failed: {meter_reason}")
        except Exception as exc:
            report.detail["meter_reason"] = str(exc)
            report.failure_reasons.append(f"meter_error: {exc}")

    report.quality_score = _quality_score(poem_text, user_prompt)
    prompt_chars = {ch for ch in user_prompt if "\u4e00" <= ch <= "\u9fff"}
    if prompt_chars and not any(ch in poem_text for ch in prompt_chars):
        report.failure_reasons.append("theme_drift: poem does not reflect prompt keywords")
    report.passed = report.format_ok and report.line_length_ok
    if PoemStructureChecker is not None:
        report.passed = report.passed and report.rhyming_ok and report.meter_ok
    return report
