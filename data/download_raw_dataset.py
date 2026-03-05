"""
Download the Complete Tang Poetry (全唐诗) dataset only.

Uses git sparse checkout to clone the chinese-poetry repo and check out
only the 全唐诗 directory. Other steps (filter, theme extraction, build)
are done by filter_poems.py, extract_themes*.py, and build_dataset.py.
"""

import os
import argparse

from .utils import download_chinese_poetry


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Download 全唐诗 (chinese-poetry, sparse checkout)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data/chinese-poetry",
        help="Target directory for the repo (default: ./data/chinese-poetry)",
    )
    args = parser.parse_args()

    tang_dir = download_chinese_poetry(args.data_dir)
    print(f"Done! 全唐诗 is at: {os.path.abspath(tang_dir)}")


if __name__ == "__main__":
    main()
