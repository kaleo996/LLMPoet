"""
Upload training dataset (train.jsonl / eval.jsonl) to Hugging Face Hub.

Uses the datasets library to load JSONL and push to a Hub repo.
Requires login: huggingface-cli login or HF_TOKEN environment variable.
"""

import os
import argparse
from datasets import DatasetDict, load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Upload train/eval JSONL to Hugging Face Hub"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data/training",
        help="Directory containing train.jsonl and eval.jsonl (default: ./data/training)",
    )
    parser.add_argument(
        "repo_id", type=str, nargs="?", default=None,
        help="Hugging Face repo id (e.g. username/nekoobasho-dataset). Required unless --dry_run.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only load and print dataset info, do not push",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private repo on the Hub",
    )
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train.jsonl")
    eval_path = os.path.join(args.data_dir, "eval.jsonl")

    if not os.path.isfile(train_path):
        print(f"Error: {train_path} not found.")
        return
    if not os.path.isfile(eval_path):
        print(f"Error: {eval_path} not found.")
        return

    print("Loading JSONL...")
    train_ds = load_dataset("json", data_files=train_path, split="train")
    eval_ds = load_dataset("json", data_files=eval_path, split="train")

    dataset = DatasetDict({
        "train": train_ds,
        "eval": eval_ds,
    })

    print(f"  train: {len(dataset['train'])} rows")
    print(f"  eval:  {len(dataset['eval'])} rows")
    print(f"  features: {dataset['train'].features}")

    if args.dry_run:
        print("\nSkipping push to Hub.")
        return

    if not args.repo_id:
        print("Error: repo_id required for upload. Example: username/nekoobasho-dataset")
        return

    print(f"\nPushing to Hugging Face Hub: {args.repo_id}")
    dataset.push_to_hub(args.repo_id, private=args.private)
    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
