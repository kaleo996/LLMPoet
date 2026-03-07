import argparse
import json

from agent.api import generate_poem_with_agent
from agent.loop import set_verbose
from model.utils import masked_poem_dict


def main():
    parser = argparse.ArgumentParser(description="LLMPoet agent-based poetry generation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./ckpt/Qwen3-8B",
        help="Local model path (default: ./ckpt/Qwen3-8B)",
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default="春天",
        help="User prompt",
    )
    parser.add_argument(
        "--poem_type",
        type=str,
        default="五言绝句",
        choices=list(masked_poem_dict.keys()),
        help="Poetry type",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional metrical variant name",
    )
    parser.add_argument(
        "--rhyme_group",
        type=str,
        default=None,
        help="Optional rhyme group name",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="simplified",
        choices=["simplified", "traditional"],
        help="Output script",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="Maximum repair rounds",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=3,
        help="Number of candidates per round",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON result",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed agent workflow",
    )
    args = parser.parse_args()

    if args.verbose:
        set_verbose(True)

    result = generate_poem_with_agent(
        user_prompt=args.user_prompt,
        poem_type=args.poem_type,
        script=args.script,
        variant=args.variant,
        rhyme_group=args.rhyme_group,
        max_rounds=args.max_rounds,
        num_candidates=args.num_candidates,
        device=args.device,
        model_path=args.model_path,
    )

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"\nUser prompt: {args.user_prompt}")
    print(f"Poetry type: {args.poem_type}")
    print(f"Success: {result.success}")
    print("-" * 50)
    print(result.poem_text)
    print("-" * 50)
    if result.evaluation_report.failure_reasons:
        print("Failure reasons:")
        for reason in result.evaluation_report.failure_reasons:
            print(f"- {reason}")


if __name__ == "__main__":
    main()
