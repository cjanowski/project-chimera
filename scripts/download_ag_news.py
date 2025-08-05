#!/usr/bin/env python3
"""
Download and cache the AG News dataset locally under ./data/ag_news.

- Uses Hugging Face datasets (no auth required).
- Saves parquet shards for train/valid/test for fast reload.
- Also writes a small README describing the layout.
- Deterministic download with a fixed HF cache directory inside project data/.

Usage:
  python scripts/download_ag_news.py --force  # re-download/overwrite
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import shutil

try:
    from datasets import load_dataset, DatasetDict
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'datasets'. Install with:\n  pip install datasets pyarrow"
    ) from e


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ag_news"
HF_LOCAL_CACHE = PROJECT_ROOT / "data" / ".hf_cache"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_readme(target_dir: Path) -> None:
    readme = target_dir / "README.md"
    content = """# AG News Dataset (cached)

This directory contains a cached copy of the AG News dataset retrieved via Hugging Face `datasets`.

Layout:
- train/*.parquet
- validation/*.parquet
- test/*.parquet

Each row contains:
- text: str  (news title + description)
- label: int (class id in [0..3])
- label_text: str (class name among {World, Sports, Business, Sci/Tech})

Regeneration:
  python scripts/download_ag_news.py

Notes:
- We store parquet shards for fast reload.
- Original source: https://huggingface.co/datasets/ag_news
"""
    readme.write_text(content, encoding="utf-8")


def to_parquet_shards(ds_dict: "DatasetDict", out_dir: Path, num_shards: int = 8) -> None:
    # Save each split as multiple parquet shards for parallel read.
    for split_name, ds in ds_dict.items():
        split_dir = out_dir / split_name
        ensure_dir(split_dir)
        # map label ids to label_text for convenience
        label_names = ds.features["label"].names if "label" in ds.features else None
        if label_names is not None and "label_text" not in ds.column_names:
            def id_to_name(example):
                idx = example["label"]
                example["label_text"] = label_names[idx]
                return example
            ds = ds.map(id_to_name)

        # Some HF datasets provide 'text' already; AG News has 'text' field (title + description concatenated).
        # Ensure we have a clean schema subset.
        keep_cols = [c for c in ["text", "label", "label_text"] if c in ds.column_names]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

        # Save to parquet shards
        ds.to_parquet(path=str(split_dir / f"{split_name}.parquet"), num_shards=num_shards)


def main(force: bool = False, shards: int = 8, subset_size: Optional[int] = None) -> None:
    ensure_dir(DATA_DIR)
    ensure_dir(HF_LOCAL_CACHE)

    # Optionally clear existing cache
    if force and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        ensure_dir(DATA_DIR)

    write_readme(DATA_DIR)

    # Respect local cache for deterministic offline-friendly behavior
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_LOCAL_CACHE))

    # Load dataset; AG News provides train/test; we'll create validation from train if missing.
    # ag_news has train (120k) and test (7.6k). We'll split a validation slice from train.
    print("Loading ag_news from Hugging Face ...")
    ds = load_dataset("ag_news")

    # Build validation split (e.g., 5% of train)
    if "validation" not in ds:
        print("Creating validation split (5% of train) ...")
        split = ds["train"].train_test_split(test_size=0.05, seed=42)
        train_ds = split["train"]
        valid_ds = split["test"]
        ds = DatasetDict(train=train_ds, validation=valid_ds, test=ds["test"])

    # Optional subsetting for very fast iteration
    if subset_size is not None:
        print(f"Subsetting each split to first {subset_size} examples ...")
        for name in list(ds.keys()):
            ds[name] = ds[name].select(range(min(subset_size, len(ds[name]))))

    print("Writing parquet shards ...")
    to_parquet_shards(ds, DATA_DIR, num_shards=shards)
    print(f"Done. Files saved under: {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing cache")
    parser.add_argument("--shards", type=int, default=8, help="Number of parquet shards per split")
    parser.add_argument("--subset", type=int, default=None, help="Optional subset size per split for quick smoke tests")
    args = parser.parse_args()
    main(force=args.force, shards=args.shards, subset_size=args.subset)