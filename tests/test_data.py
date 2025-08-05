import pytest
import torch

from src.project_chimera.data.preprocess import TokenizerConfig, build_tokenizer, CausalTextDataset, build_dataloaders


def test_tokenizer_and_dataset_shapes():
    cfg = TokenizerConfig(pretrained_name="gpt2", lowercase=True, max_length=128)
    tok = build_tokenizer(cfg)

    # small subset for speed
    ds = CausalTextDataset("train", tok, cfg, limit=8)
    assert len(ds) == 8

    sample = ds[0]
    assert "input_ids" in sample and "attention_mask" in sample and "labels" in sample
    assert sample["input_ids"].shape[0] == cfg.max_length
    assert sample["attention_mask"].shape[0] == cfg.max_length
    assert sample["labels"].shape[0] == cfg.max_length

    # Ensure labels ignore index at pad positions if any
    pad_id = tok.pad_token_id
    if pad_id is not None:
        mask = sample["input_ids"] == pad_id
        if mask.any():
            assert torch.all(sample["labels"][mask] == -100)


def test_dataloaders_batching():
    cfg = TokenizerConfig(pretrained_name="gpt2", lowercase=True, max_length=64)
    tok = build_tokenizer(cfg)

    train_loader, val_loader = build_dataloaders(tok, cfg, train_limit=32, val_limit=16, batch_size=8)
    xb = next(iter(train_loader))
    assert xb["input_ids"].shape == (8, 64)
    assert xb["attention_mask"].shape == (8, 64)
    assert xb["labels"].shape == (8, 64)