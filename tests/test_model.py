import torch

from src.project_chimera.baseline import BaselineConfig, GPTDecoder
from src.project_chimera.data.preprocess import TokenizerConfig, build_tokenizer, CausalTextDataset


def test_model_forward_shapes():
    # Build tokenizer to get vocab size consistent with preprocessing
    tok_cfg = TokenizerConfig(pretrained_name="gpt2", lowercase=True, max_length=32)
    tokenizer = build_tokenizer(tok_cfg)
    vocab_size = tokenizer.vocab_size

    # Small model for unit test speed
    cfg = BaselineConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        ff_dim=256,
        dropout=0.0,
        max_seq_len=tok_cfg.max_length,
        tie_weights=True,
    )
    model = GPTDecoder(cfg)
    model.eval()

    # Dummy batch from real dataset encoding to ensure compatibility
    ds = CausalTextDataset("train", tokenizer, tok_cfg, limit=4)
    batch = {
        "input_ids": torch.stack([ds[i]["input_ids"] for i in range(4)], dim=0),
        "attention_mask": torch.stack([ds[i]["attention_mask"] for i in range(4)], dim=0),
        "labels": torch.stack([ds[i]["labels"] for i in range(4)], dim=0),
    }

    with torch.no_grad():
        out = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

    assert "logits" in out and "loss" in out
    B, T = batch["input_ids"].shape
    assert out["logits"].shape == (B, T, vocab_size)
    assert out["loss"].ndim == 0


def test_autoregressive_masking_no_nan_inf():
    tok_cfg = TokenizerConfig(pretrained_name="gpt2", lowercase=True, max_length=16)
    tokenizer = build_tokenizer(tok_cfg)
    vocab_size = tokenizer.vocab_size

    cfg = BaselineConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=1,
        n_heads=4,
        ff_dim=128,
        dropout=0.0,
        max_seq_len=tok_cfg.max_length,
        tie_weights=True,
    )
    model = GPTDecoder(cfg)
    model.train()

    ds = CausalTextDataset("train", tokenizer, tok_cfg, limit=8)
    batch = {
        "input_ids": torch.stack([ds[i]["input_ids"] for i in range(8)], dim=0),
        "attention_mask": torch.stack([ds[i]["attention_mask"] for i in range(8)], dim=0),
        "labels": torch.stack([ds[i]["labels"] for i in range(8)], dim=0),
    }

    out = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    loss = out["loss"]
    assert torch.isfinite(loss), "Loss should be finite during forward/backward smoke test"