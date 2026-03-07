"""
Step 1 of CharPoet-style evaluation for the agent pipeline: batch generate
poems from idiom and instruction theme files, save to eval_poems_agent.json
for downstream format/rhyme stats and content quality scoring.
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

from agent.api import generate_poem_with_agent
from agent.loop import set_verbose
from model.utils import masked_poem_dict


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
        description="Generate poems for evaluation from idiom and instruction theme files via the agent pipeline."
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
        default=os.path.join(_PROJECT_ROOT, "eval", "eval_poems_agent.json"),
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
        "--model_path",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "ckpt", "Qwen3-8B"),
        help="Local model path.",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="Maximum repair rounds for the agent.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=3,
        help="Number of draft candidates per round.",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="simplified",
        choices=["simplified", "traditional"],
        help="Output script.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed agent workflow",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include agent generation trace in the output JSON.",
    )
    args = parser.parse_args()

    if args.verbose:
        set_verbose(True)

    idioms = load_theme_lines(args.idioms)
    instructions = load_theme_lines(args.instructions)
    if args.limit is not None:
        idioms = idioms[: args.limit]
        instructions = instructions[: args.limit]

    print(f"Loaded {len(idioms)} idioms, {len(instructions)} instructions.")
    poem_types = list(masked_poem_dict.keys())
    print(f"Poem types: {poem_types}")
    print("Using agent pipeline with local model runtime...")

    results = {"idiom": [], "instruction": []}
    total = (len(idioms) + len(instructions)) * len(poem_types)

    with tqdm(total=total, desc="Generating poems (agent)") as pbar:
        for theme_type, themes, key in [
            ("keyword", idioms, "idiom"),
            ("instruction", instructions, "instruction"),
        ]:
            for theme in themes:
                for poem_type in poem_types:
                    agent_result = None
                    try:
                        agent_result = generate_poem_with_agent(
                            user_prompt=theme,
                            poem_type=poem_type,
                            script=args.script,
                            max_rounds=args.max_rounds,
                            num_candidates=args.num_candidates,
                            device=args.device,
                            model_path=args.model_path,
                        )
                        poem_text = agent_result.poem_text
                        evaluation_report = agent_result.evaluation_report.to_dict()
                        success = agent_result.success
                    except Exception as e:
                        tqdm.write(
                            f"Error generating poem for {theme_type} theme={theme!r} poem_type={poem_type}: {e}"
                        )
                        poem_text = ""
                        evaluation_report = {}
                        success = False

                    record = {
                        "theme": theme,
                        "theme_type": theme_type,
                        "poem_type": poem_type,
                        "user_prompt": theme,
                        "poem_text": poem_text,
                        "success": success,
                        "evaluation_report": evaluation_report,
                    }
                    if args.debug:
                        record["generation_trace"] = [
                            attempt.to_dict() for attempt in agent_result.attempts
                        ] if agent_result is not None else []
                    results[key].append(record)
                    pbar.update(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(
        f"Saved {len(results['idiom'])} + {len(results['instruction'])} records to {args.output}"
    )


if __name__ == "__main__":
    main()
