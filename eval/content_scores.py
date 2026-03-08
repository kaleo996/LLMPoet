"""
Step 3 of CharPoet-style evaluation: use Moonshot API to score each poem on
five content quality dimensions (Fluency, Meaning, Coherence, Relevance, Aesthetics)
on a 1-5 scale. Reads eval_poems.json, writes content_scores.json and summary.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI


# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Five criteria from CharPoet §5.4 (1-5 scale)
CRITERIA = {
    "Fluency": "Does the poem obey the grammatical, structural and phonological rules? (语法、结构与音韵规则是否合理)",
    "Meaning": "Does the poem convey some certain messages? (是否传达明确意涵)",
    "Coherence": "Is the poem as a whole coherent in meaning and theme? (整体意义与主题是否连贯)",
    "Relevance": "Does the poem express user topics well? (是否贴合用户主题)",
    "Aesthetics": "Does the poem have some poetic and artistic beauties? (是否具有诗意与艺术美感)",
}


def get_client():
    """Build OpenAI-compatible client for Moonshot."""
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise ValueError(
            "Set MOONSHOT_API_KEY in the environment. "
            "Get your key from https://platform.moonshot.ai"
        )
    base_url = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")
    # Timeout so stuck API calls return
    timeout = 60.0
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def build_scoring_prompt(poem_text: str, user_prompt: str, poem_type: str) -> str:
    """Build the prompt for the LLM to score the poem."""
    criteria_text = "\n".join(
        f"- {name} (1-5): {desc}" for name, desc in CRITERIA.items()
    )
    return f"""请针对下面这首古诗，根据其具体内容在五个维度上分别打 1-5 分（5 为最好）。请严格使用 1-5 全区间：若某维度有明显不足请打 1 至 3 分，不要全部给高分。

用户主题/指令：{user_prompt}
诗体：{poem_type}

诗歌正文：
{poem_text}

评分维度：
{criteria_text}

请先逐维度用一句话说明评分理由，最后单独一行只输出一个 JSON 对象。键为英文维度名，值为 1-5 的整数。例如最后一行：
{{"Fluency": 4, "Meaning": 5, "Coherence": 4, "Relevance": 5, "Aesthetics": 4}}
"""


def has_valid_scores(rec: dict) -> bool:
    """Return True if record has valid 1-5 scores for all five criteria."""
    for key in CRITERIA:
        v = rec.get(key)
        if v is None or not (1 <= v <= 5):
            return False
    return True


def parse_scores(response_text: str) -> dict | None:
    """Extract five scores from model response. Returns dict or None on failure."""
    text = response_text.strip()
    # Try to find a JSON object (may be last line after reasoning)
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    result = {}
    for key in CRITERIA:
        if key in obj and isinstance(obj[key], (int, float)):
            v = int(obj[key])
            if 1 <= v <= 5:
                result[key] = v
            else:
                result[key] = None
        else:
            result[key] = None
    return result if len(result) == 5 else None


def load_poems(path: str) -> list[dict]:
    """Load all poem records (idiom + instruction) from eval_poems.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    out.extend(data.get("idiom", []))
    out.extend(data.get("instruction", []))
    return out


class RateLimiter:
    """Allow at most rpm requests per minute; call wait() before each request start."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next_available = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_available:
                time.sleep(self._next_available - now)
                now = time.monotonic()
            self._next_available = now + self._interval


def _score_one(
    client: OpenAI,
    model: str,
    temperature: float,
    rate_limiter: RateLimiter | None,
    idx: int,
    rec: dict,
) -> tuple[int, dict]:
    """Call API for one poem; return (idx, row)."""
    theme = rec.get("theme", "")
    poem_type = rec.get("poem_type", "")
    theme_type = rec.get("theme_type", "")
    poem_text = rec.get("poem_text", "")
    user_prompt = rec.get("user_prompt", theme)
    if rate_limiter is not None:
        rate_limiter.wait()
    prompt = build_scoring_prompt(poem_text, user_prompt, poem_type)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        scores = parse_scores(content)
    except Exception as e:
        print(f"Error: {theme_type} theme={theme!r} poem_type={poem_type}: {e}")
        scores = None
    if scores is None:
        print(
            f"Warning: could not parse scores (theme={theme!r}, poem_type={poem_type}, theme_type={theme_type})"
        )
        scores = {k: None for k in CRITERIA}
    row = {
        "theme": theme,
        "theme_type": theme_type,
        "poem_type": poem_type,
        "user_prompt": user_prompt,
        "poem_text": poem_text,
        **scores,
    }
    return (idx, row)


def main():
    parser = argparse.ArgumentParser(
        description="Score poems on five content quality dimensions using Moonshot API."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "eval", "eval_poems.json"),
        help="Path to eval_poems.json from stage 1.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "eval", "content_scores.json"),
        help="Path to write content_scores.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="moonshot-v1-8k",
        help="Moonshot model name (e.g. moonshot-v1-8k, moonshot-v1-32k).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: max number of poems to score (for debugging).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip poems already present in --output (resume from previous run).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls when --workers=1 (ignored if workers>1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Max concurrent API requests (match API concurrency limit, e.g. 3).",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=20,
        help="Rate limit: max requests per minute (e.g. 20). Used when workers>1.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for API (0=deterministic; 0.2-0.5 may improve score variation).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: {args.input} not found. Run eval/generate_for_eval.py first.")
        sys.exit(1)

    poems = load_poems(args.input)
    if not poems:
        print("No poems in input. Exiting.")
        return

    if args.limit is not None:
        poems = poems[: args.limit]

    existing = {}
    if args.resume and os.path.isfile(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                data = json.load(f)
            for rec in data.get("scores", []):
                key = (rec.get("theme"), rec.get("poem_type"), rec.get("theme_type"))
                existing[key] = rec
        except Exception:
            pass

    client = get_client()
    # Results by index so we can merge existing + API and keep poem order
    results_by_idx: list[dict | None] = [None] * len(poems)
    to_score: list[tuple[int, dict]] = []
    for idx, rec in enumerate(poems):
        theme = rec.get("theme", "")
        poem_type = rec.get("poem_type", "")
        theme_type = rec.get("theme_type", "")
        key = (theme, poem_type, theme_type)
        if key in existing and has_valid_scores(existing[key]):
            results_by_idx[idx] = existing[key]
        else:
            to_score.append((idx, rec))

    if args.resume and to_score:
        print(f"Resume: {len(poems) - len(to_score)} already have valid scores, {len(to_score)} to rescore.")

    if args.workers <= 1:
        # Sequential: original behavior with --delay
        rate_limiter = None
        for idx, rec in tqdm(to_score, desc="Scoring poems"):
            _, row = _score_one(
                client, args.model, args.temperature, rate_limiter, idx, rec
            )
            results_by_idx[idx] = row
            time.sleep(args.delay)
    else:
        # Concurrent: respect --rpm (stagger request starts)
        rate_limiter = RateLimiter(args.rpm)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _score_one,
                    client,
                    args.model,
                    args.temperature,
                    rate_limiter,
                    idx,
                    rec,
                ): idx
                for idx, rec in to_score
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring poems"):
                idx, row = future.result()
                results_by_idx[idx] = row
    results = list(results_by_idx)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"scores": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} records to {args.output}")

    # Summary: mean per dimension by theme_type
    for dim in CRITERIA:
        by_theme = defaultdict(list)
        for row in results:
            v = row.get(dim)
            if v is not None:
                by_theme[row["theme_type"]].append(v)
        print(f"\n  {dim}:")
        for tt in ("keyword", "instruction"):
            vals = by_theme.get(tt, [])
            if vals:
                print(f"    {tt}: mean={sum(vals)/len(vals):.2f} n={len(vals)}")
            else:
                print(f"    {tt}: (no valid scores)")


if __name__ == "__main__":
    main()
