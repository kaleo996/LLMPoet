"""
Step 1 of CharPoet-style evaluation: batch generate poems from idiom and
instruction theme files, save to eval_poems.json for format/rhyme stats and
content quality scoring.
"""
import argparse
import json
import os
import sys

from tqdm import tqdm

# Project root (parent of eval/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.generation import load_token_free_model, generate_poems_batch
from model.utils import POEM_STRUCTURE


def load_theme_lines(path: str, max_lines: int = 100) -> list[str]:
    """Load non-empty, non-comment lines from a theme file (UTF-8)."""
    if not os.path.isfile(path):
        return []
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
            if len(lines) >= max_lines:
                break
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate poems for evaluation from idiom and instruction theme files."
    )
    parser.add_argument(
        "--idioms",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "eval", "eval_idioms.txt"),
        help="Path to idioms file (one per line).",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "eval", "eval_instructions.txt"),
        help="Path to instructions file (one per line).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "eval", "eval_poems.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: max number of themes per file (for debugging).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for parallel poem generation.",
    )
    args = parser.parse_args()

    idioms = load_theme_lines(args.idioms)
    instructions = load_theme_lines(args.instructions)
    if args.limit is not None:
        idioms = idioms[: args.limit]
        instructions = instructions[: args.limit]

    print(f"Loaded {len(idioms)} idioms, {len(instructions)} instructions.")
    poem_types = list(POEM_STRUCTURE.keys())
    print(f"Poem types: {poem_types}")

    print("Loading model...")
    model, tokenizer = load_token_free_model(device=args.device)

    results = {"idiom": [], "instruction": []}
    total = (len(idioms) + len(instructions)) * len(poem_types)

    with tqdm(total=total, desc="Generating poems") as pbar:
        for theme_type, themes, key in [
            ("keyword", idioms, "idiom"),
            ("instruction", instructions, "instruction"),
        ]:
            for poem_type in poem_types:
                for start in range(0, len(themes), args.batch_size):
                    batch_themes = themes[start : start + args.batch_size]
                    retry = 0
                    while True:
                        retry += 1
                        try:
                            batch_poems = generate_poems_batch(
                                model=model,
                                tokenizer=tokenizer,
                                user_prompts=batch_themes,
                                poem_type=poem_type,
                                device=args.device,
                                script="simplified",
                            )
                            if len(batch_poems) != len(batch_themes):
                                raise RuntimeError(
                                    f"Batch output size mismatch: {len(batch_poems)} vs {len(batch_themes)}"
                                )
                            break
                        except Exception as e:
                            tqdm.write(
                                f"Batch generation failed for {theme_type} poem_type={poem_type} "
                                f"start={start} size={len(batch_themes)} retry={retry}: {e}. Retrying..."
                            )

                    for theme, poem_text in zip(batch_themes, batch_poems):
                        results[key].append({
                            "theme": theme,
                            "theme_type": theme_type,
                            "poem_type": poem_type,
                            "user_prompt": theme,
                            "poem_text": poem_text,
                        })
                        pbar.update(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results['idiom'])} + {len(results['instruction'])} records to {args.output}")


if __name__ == "__main__":
    main()
