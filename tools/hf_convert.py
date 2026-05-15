#!/usr/bin/env python3
"""Convert the CodingJEPA corpus to a HuggingFace DatasetDict and push to the Hub.

Usage:
    python tools/hf_convert.py --pairs data/pairs/train.parquet ... \\
        --repo CodingJEPA/human-python-refactors
    python tools/hf_convert.py --pairs data/pairs/train.parquet --dry-run

Converts pairs/*.parquet + chunks/*.parquet into a DatasetDict with two configs:
  - "pairs": train/val/test splits of refactoring pairs
  - "chunks": the raw code chunks

Requires: datasets, huggingface_hub
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/pairs"),
        help="Directory containing train/val/test parquet files",
    )
    p.add_argument(
        "--chunks-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/chunks"),
        help="Directory containing chunk parquet files",
    )
    p.add_argument(
        "--repo",
        default="CodingJEPA/human-python-refactors",
        help="HF Hub repository ID",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the DatasetDict but do not push to Hub",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the HF repo as private",
    )
    return p


def build_dataset_dict(pairs_dir: pathlib.Path, chunks_dir: pathlib.Path) -> object:
    """Load parquet files and build a DatasetDict."""
    try:
        import pandas as pd  # type: ignore[import-untyped,unused-ignore]
        from datasets import Dataset, DatasetDict  # type: ignore[import-not-found,unused-ignore]
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}. Run: pip install datasets pandas")

    splits = {}
    for split in ("train", "val", "test"):
        parquet = pairs_dir / f"{split}.parquet"
        if parquet.exists():
            splits[split] = Dataset.from_pandas(pd.read_parquet(parquet))

    if not splits:
        sys.exit(f"No parquet files found in {pairs_dir}")

    return DatasetDict(splits)


def main() -> None:
    args = build_parser().parse_args()
    print(f"Building DatasetDict from {args.pairs_dir} ...")
    dd = build_dataset_dict(args.pairs_dir, args.chunks_dir)
    print(dd)

    if args.dry_run:
        print("Dry run: not pushing to Hub.")
        return

    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found,unused-ignore]
    except ImportError as exc:
        sys.exit(f"Missing dependency: {exc}. Run: pip install huggingface-hub")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    dd.push_to_hub(args.repo)  # type: ignore[attr-defined]
    print(f"Pushed to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
