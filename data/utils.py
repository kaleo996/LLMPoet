import os
import glob
import json
import subprocess
from typing import List, Dict

# Data download
def download_chinese_poetry(target_dir: str) -> str:
    """Download chinese-poetry repo via git sparse checkout (全唐诗 only).

    Returns:
        Path to the 全唐诗 directory.
    """
    tang_dir = os.path.join(target_dir, "全唐诗")
    if os.path.isdir(tang_dir) and any(f.endswith(".json") for f in os.listdir(tang_dir)):
        print(f"[download] 全唐诗 already exists at {tang_dir}, skipping download.")
        return tang_dir

    print(f"[download] Cloning chinese-poetry repo (sparse checkout) into {target_dir} ...")
    os.makedirs(os.path.dirname(target_dir) or ".", exist_ok=True)

    # Sparse clone: only metadata, no blobs yet
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "--filter=blob:none", "--sparse",
            "https://github.com/chinese-poetry/chinese-poetry.git",
            target_dir,
        ],
        check=True,
    )
    # Check out only the 全唐诗 folder
    subprocess.run(
        ["git", "sparse-checkout", "set", "全唐诗"],
        cwd=target_dir,
        check=True,
    )
    print(f"[download] Done. Tang poetry files at {tang_dir}")
    return tang_dir


def load_tang_poems(tang_dir: str) -> List[Dict]:
    """Load all poem objects from 全唐诗 JSON files."""
    json_files = sorted(glob.glob(os.path.join(tang_dir, "poet.tang.*.json")))
    if not json_files:
        # Fallback: try any JSON file in the directory
        json_files = sorted(glob.glob(os.path.join(tang_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {tang_dir}")

    poems = []
    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                poems.extend(data)
            elif isinstance(data, dict):
                poems.append(data)
    print(f"[load] Loaded {len(poems)} raw poems from {len(json_files)} files.")
    return poems
