#!/usr/bin/env python3
"""
Smoke test for the AG News data pipeline.

This script:
1) Ensures AG News parquet data exists (prompts to run download script if missing).
2) Builds tokenizer and DataLoaders using src/project_chimera/data/ag_news.py
3) Iterates a few batches to validate shapes and throughput basics.

Requirements:
  pip install datasets pyarrow transformers torch

Usage:
  python scripts/smoke_test_data_pipeline.py --data_dir data/ag_news --model_name bert-base-uncased --max_len 128 --batch_size 16 --num_workers 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import torch

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.project_chimera.data.ag_news import (  # noqa: E402
    TokenizerConfig,
    get_tokenizer,
    build_datasets_and_loaders,
)


def ensure_data_present(data_dir: Path) -> None:
    train = data_dir / "train"
    val = data_dir / "validation"
    test = data_dir / "test"
    if not (train.exists() and val.exists() and test.exists()):
        raise SystemExit(
            f"AG News parquet not found under {data_dir}. "
            "Run: python scripts/download_ag_news.py --subset 2000"
        )
    # Spot check at least one parquet file
    if not any(train.glob("*.parquet")):
        raise SystemExit(
            f"No parquet files in {train}. Re-run download: python scripts/download_ag_news.py"
        )


def main(
    data_dir: Path,
    model_name: str,
    max_len: int,
    batch_size: int,
    num_workers: int,
    device: str,
):
    ensure_data_present(data_dir)

    tok_cfg = TokenizerConfig(model_name=model_name, max_length=max_len)
    tokenizer = get_tokenizer(tok_cfg)

    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = build_datasets_and_loaders(
        data_root=data_dir,
        tokenizer=tokenizer,
        tok_cfg=tok_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Dataset sizes | train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # Iterate a few batches and check tensor shapes/types
    def sample_iter(name, dl, steps=3):
        print(f"\nIterating {name} loader for {steps} steps ...")
        t0 = time.time()
        seen = 0
        for i, batch in enumerate(dl):
            if i >= steps:
                break
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attn = batch.get("attention_mask", None)

            assert isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2
            assert isinstance(labels, torch.Tensor) and labels.dim() == 1
            if attn is not None:
                assert attn.shape == input_ids.shape

            print(
                f"  step {i}: input_ids {tuple(input_ids.shape)} "
                f"labels {tuple(labels.shape)}"
                + (f" attn {tuple(attn.shape)}" if attn is not None else "")
            )
            seen += input_ids.size(0)
        dt = time.time() - t0
        if dt > 0:
            print(f"Throughput (approx): {seen / dt:.2f} samples/s")

    sample_iter("train", train_dl)
    sample_iter("val", val_dl)
    sample_iter("test", test_dl)

    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ag_news")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        data_dir=Path(args.data_dir),
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )