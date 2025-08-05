#!/usr/bin/env python3
import argparse
from dataclasses import asdict

import torch

from src.project_chimera.data.preprocess import (
    TokenizerConfig,
    build_tokenizer,
    build_dataloaders,
)
from src.project_chimera.baseline import BaselineConfig, GPTDecoder
from src.project_chimera.trainer import BaselineTrainer, TrainConfig


def parse_args():
    p = argparse.ArgumentParser(description="Smoke train baseline GPT-style decoder on AG News subset")
    p.add_argument("--model_name", type=str, default="gpt2", help="HF tokenizer name")
    p.add_argument("--lowercase", action="store_true", default=True, help="Lowercase text before tokenization")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--train_limit", type=int, default=2048)
    p.add_argument("--val_limit", type=int, default=512)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    return p.parse_args()


def main():
    args = parse_args()

    # Tokenizer setup
    tok_cfg = TokenizerConfig(
        pretrained_name=args.model_name,
        lowercase=args.lowercase,
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    tokenizer = build_tokenizer(tok_cfg)

    # Data
    train_loader, val_loader = build_dataloaders(
        tokenizer,
        tok_cfg,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        batch_size=args.batch_size,
    )

    # Model
    vocab_size = tokenizer.vocab_size
    model_cfg = BaselineConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_seq_len=args.max_length,
        tie_weights=True,
    )
    model = GPTDecoder(model_cfg)

    # Trainer
    train_cfg = TrainConfig(
        lr=args.lr,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        amp=True,
    )
    trainer = BaselineTrainer(model, train_loader, val_loader, train_cfg)

    final_metrics = trainer.train()
    print("FINAL_METRICS", final_metrics)


if __name__ == "__main__":
    main()