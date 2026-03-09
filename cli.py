import argparse

from model.generation import load_token_free_model, generate_poem


def main():
    parser = argparse.ArgumentParser(description="NekooBasho poetry generation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./ckpt/Qwen3-8B",
        help="Local model path (default: ./ckpt/Qwen3-8B)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./ckpt/single_char_tokens.json",
        help="Token-free config file path (default: ./ckpt/single_char_tokens.json)",
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default="Write a poem about spring",
        help="User prompt",
    )
    parser.add_argument(
        "--poem_type",
        type=str,
        default="五言绝句",
        help="Poetry type (e.g., 五言绝句, 七言律诗, ...)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Metrical pattern variant name (e.g., 仄起首句不入韵). If omitted, a random variant is chosen.",
    )
    parser.add_argument(
        "--rhyme_group",
        type=str,
        default=None,
        help="Pingshui rhyme group name (e.g., 上平聲一東, 下平聲七陽). "
        "If omitted, the rhyme group is auto-detected from the first rhyme position generated.",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="simplified",
        choices=["simplified", "traditional"],
        help="Output script: simplified or traditional Chinese (default: simplified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    model, tokenizer = load_token_free_model(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
    )

    print(f"\nUser prompt: {args.user_prompt}")
    print(f"Poetry type: {args.poem_type}")
    print(f"Script: {args.script}")

    generated_poem = generate_poem(
        model=model,
        tokenizer=tokenizer,
        user_prompt=args.user_prompt,
        poem_type=args.poem_type,
        variant=args.variant,
        rhyme_group=args.rhyme_group,
        script=args.script,
        device=args.device,
    )

    print("-" * 50)
    print("\nGenerated poetry:")
    print(generated_poem)
    print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
