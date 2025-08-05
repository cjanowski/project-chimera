from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'pyarrow'. Install with:\n  pip install pyarrow"
    ) from e

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError as e:
    raise SystemExit(
        "Missing dependency 'transformers'. Install with:\n  pip install transformers"
    ) from e


# ------------------------------
# Tokenization
# ------------------------------

@dataclass
class TokenizerConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    truncation: bool = True
    padding: str = "max_length"  # 'longest' at collate time for efficiency; keep here for single encode.
    add_special_tokens: bool = True


def get_tokenizer(cfg: TokenizerConfig) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    # Ensure pad token if missing (e.g., for GPT2-like tokenizers)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    return tok


# ------------------------------
# Dataset from Parquet shards
# ------------------------------

class ParquetTextClassificationDataset(Dataset):
    """
    Lightweight Dataset to stream rows from a directory of parquet shards.
    Expects schema with columns: 'text' (str), 'label' (int). 'label_text' is optional.
    """

    def __init__(self, split_dir: Path):
        self.split_dir = Path(split_dir)
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Accept any parquet files in the directory
        self.files: List[Path] = sorted(self.split_dir.glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No parquet files in: {self.split_dir}")

        # Read metadata to compute total rows and mapping from global idx to (file, row)
        self._file_row_counts: List[int] = []
        self._cumulative: List[int] = []
        total = 0
        for f in self.files:
            md = pq.ParquetFile(str(f)).metadata
            # Sum row counts of all row groups
            rows = sum(md.row_group(i).num_rows for i in range(md.num_row_groups))
            self._file_row_counts.append(rows)
            total += rows
            self._cumulative.append(total)
        self._total_rows = total

    def __len__(self) -> int:
        return self._total_rows

    def _locate(self, idx: int) -> Tuple[int, int]:
        # Find which file contains the row for global idx
        # Binary search over cumulative
        lo, hi = 0, len(self._cumulative) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if idx < self._cumulative[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        file_idx = lo
        prev_cum = self._cumulative[file_idx - 1] if file_idx > 0 else 0
        row_in_file = idx - prev_cum
        return file_idx, row_in_file

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(idx)
        file_idx, row_in_file = self._locate(idx)
        fpath = self.files[file_idx]
        pf = pq.ParquetFile(str(fpath))
        # Identify which row group the row falls into, then read just that group
        acc = 0
        target_group = 0
        for g in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            if row_in_file < acc + n:
                target_group = g
                break
            acc += n
        tbl: pa.Table = pf.read_row_group(target_group, columns=["text", "label"])
        # row within the row group
        local_row = row_in_file - acc
        text = tbl.column("text")[local_row].as_py()
        label = tbl.column("label")[local_row].as_py()
        return {"text": text, "label": int(label)}


# ------------------------------
# Collator
# ------------------------------

@dataclass
class CollatorConfig:
    pad_to_multiple_of: Optional[int] = 8  # good for tensor cores when using AMP
    return_attention_mask: bool = True


class TextClassificationCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, tok_cfg: TokenizerConfig, cfg: CollatorConfig):
        self.tok = tokenizer
        self.tok_cfg = tok_cfg
        self.cfg = cfg

    def __call__(self, batch: List[dict]):
        texts = [ex["text"] for ex in batch]
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)

        encoded = self.tok(
            texts,
            padding="longest",  # pad dynamically per batch
            truncation=self.tok_cfg.truncation,
            max_length=self.tok_cfg.max_length,
            add_special_tokens=self.tok_cfg.add_special_tokens,
            return_tensors="pt",
            return_attention_mask=self.cfg.return_attention_mask,
            pad_to_multiple_of=self.cfg.pad_to_multiple_of,
        )
        # Standardize field names for downstream
        out = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded.get("attention_mask", None),
            "labels": labels,
        }
        return out


# ------------------------------
# Builder
# ------------------------------

@dataclass
class DataConfig:
    data_root: Path
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


def build_datasets_and_loaders(
    data_root: Path,
    tokenizer: PreTrainedTokenizerBase,
    tok_cfg: TokenizerConfig,
    collate_cfg: Optional[CollatorConfig] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
    """
    Build train/valid/test datasets and dataloaders for AG News parquet layout produced by scripts/download_ag_news.py.
    """
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "validation"
    test_dir = data_root / "test"

    train_ds = ParquetTextClassificationDataset(train_dir)
    val_ds = ParquetTextClassificationDataset(val_dir)
    test_ds = ParquetTextClassificationDataset(test_dir)

    if collate_cfg is None:
        collate_cfg = CollatorConfig()

    collator = TextClassificationCollator(tokenizer, tok_cfg, collate_cfg)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collator,
        drop_last=False,
        shuffle=False,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, **common)
    test_loader = DataLoader(test_ds, **common)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ------------------------------
# Convenience CLI smoke test
# ------------------------------

def _len_and_batch_shapes(dl: DataLoader, n: int = 2) -> List[Tuple[Tuple[int, int], int]]:
    out = []
    it = iter(dl)
    for _ in range(n):
        batch = next(it)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        out.append(((input_ids.size(0), input_ids.size(1)), int(labels.size(0))))
    return out


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ag_news", help="Path to ag_news data root")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="HF tokenizer model name")
    parser.add_argument("--max_len", type=int, default=128, help="Max tokenized sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    args = parser.parse_args()

    tok_cfg = TokenizerConfig(model_name=args.model_name, max_length=args.max_len)
    tokenizer = get_tokenizer(tok_cfg)

    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = build_datasets_and_loaders(
        data_root=Path(args.data_dir),
        tokenizer=tokenizer,
        tok_cfg=tok_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Dataset sizes:", len(train_ds), len(val_ds), len(test_ds))
    for name, dl in [("train", train_dl), ("val", val_dl), ("test", test_dl)]:
        shapes = _len_and_batch_shapes(dl, n=2)
        print(f"{name} sample batch shapes:", shapes)