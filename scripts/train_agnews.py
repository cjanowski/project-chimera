#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

import torch

from project_chimera.baseline import BaselineConfig, GPTDecoder
from project_chimera.trainer import BaselineTrainer, TrainConfig as TrainerConfig
from project_chimera.data.preprocess import (
    TokenizerConfig as TokCfg,
    build_tokenizer,
    build_dataloaders,
)
from project_chimera.utils.device import device_name, get_device
from project_chimera.utils.repro import set_seed

# MPS-friendly defaults: disable tokenizers parallelism to avoid fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Prefer higher matmul precision where supported (has effect on MPS/CPU backends in recent PyTorch)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


@dataclass
class ExpConfig:
    # Data/tokenizer
    model_name: str = "gpt2"
    lowercase: bool = True
    max_length: int = 128
    train_limit: int = 2048
    val_limit: int = 512
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    data_root: str = "data/ag_news"

    # Model - Reduced capacity to prevent overfitting
    d_model: int = 256
    n_layers: int = 2  # Reduced from 4 to 2
    n_heads: int = 4
    ff_dim: int = 512  # Reduced from 1024 to 512
    dropout: float = 0.4  # Increased from 0.1 to 0.4
    max_seq_len: int = 128
    tie_weights: bool = True

    # MoE toggles
    moe_enabled: bool = False
    moe_n_experts: int = 4
    moe_top_k: int = 1
    moe_activation: str = "gelu"
    moe_noisy_gate: bool = False

    # Optim/training - Anti-overfitting configuration
    lr: float = 1e-4  # Reduced from 3e-4 to 1e-4
    weight_decay: float = 0.1  # Increased from 0.01 to 0.1
    max_steps: int = 1000  # Increased to allow for early stopping
    log_every: int = 20
    eval_every: int = 100
    grad_clip: float = 1.0
    amp: bool = True

    # Misc
    seed: int = 42
    runs_dir: str = "runs"
    run_tag: str = "agnews"


def build_model_and_loaders(cfg: ExpConfig):
    # Seed and device
    set_seed(cfg.seed, deterministic=True)
    device = get_device()
    print(f"Using device: {device} ({device_name(device)})")

    # Tokenizer and loaders
    tok = build_tokenizer(TokCfg(pretrained_name=cfg.model_name, lowercase=cfg.lowercase, max_length=cfg.max_length))

    # On MPS, pin_memory is not supported and produces a warning; force-disable for dataloaders
    effective_pin_memory = cfg.pin_memory and (device.type != "mps")

    train_loader, val_loader = build_dataloaders(
        tok,
        TokCfg(pretrained_name=cfg.model_name, lowercase=cfg.lowercase, max_length=cfg.max_length),
        train_limit=cfg.train_limit,
        val_limit=cfg.val_limit,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=effective_pin_memory,
        data_root=cfg.data_root,
    )

    # Model config
    model_cfg = BaselineConfig(
        vocab_size=tok.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        tie_weights=cfg.tie_weights,
        moe_enabled=cfg.moe_enabled,
        moe_n_experts=cfg.moe_n_experts,
        moe_top_k=cfg.moe_top_k,
        moe_activation=cfg.moe_activation,
        moe_noisy_gate=cfg.moe_noisy_gate,
    )
    model = GPTDecoder(model_cfg)
    trainer_cfg = TrainerConfig(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_steps=cfg.max_steps,
        log_every=cfg.log_every,
        eval_every=cfg.eval_every,
        grad_clip=cfg.grad_clip,
        amp=(cfg.amp and device.type == "cuda"),
    )
    trainer = BaselineTrainer(model, train_loader, val_loader, trainer_cfg, device=device)
    return trainer, tok


def run_one(cfg: ExpConfig, variant: str) -> Dict[str, Any]:
    # Set variant toggles
    is_dense = variant == "dense"
    is_moe = variant == "moe"
    assert is_dense or is_moe, f"Unknown variant: {variant}"

    local_cfg = ExpConfig(**asdict(cfg))
    if is_dense:
        local_cfg.moe_enabled = False
    else:
        local_cfg.moe_enabled = True

    trainer, tok = build_model_and_loaders(local_cfg)
    final_metrics = trainer.train()

    out = {
        "variant": variant,
        "final_metrics": final_metrics,
        "config": asdict(local_cfg),
        "vocab_size": tok.vocab_size,
        "device": device_name(),
    }
    return out


def save_results(results: Dict[str, Any], cfg: ExpConfig, suffix: str) -> Path:
    runs_dir = Path(cfg.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{cfg.run_tag}_{suffix}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to: {path}")
    return path


def main():
    p = argparse.ArgumentParser(description="Train dense and MoE variants on AG News (short runs).")
    # Data/tokenizer
    p.add_argument("--model_name", type=str, default=ExpConfig.model_name)
    p.add_argument("--lowercase", action="store_true", default=ExpConfig.lowercase)
    p.add_argument("--max_length", type=int, default=ExpConfig.max_length)
    p.add_argument("--train_limit", type=int, default=ExpConfig.train_limit)
    p.add_argument("--val_limit", type=int, default=ExpConfig.val_limit)
    p.add_argument("--batch_size", type=int, default=ExpConfig.batch_size)
    p.add_argument("--num_workers", type=int, default=ExpConfig.num_workers)
    p.add_argument("--no_pin_memory", action="store_true", help="Disable pin_memory")
    p.add_argument("--data_root", type=str, default=ExpConfig.data_root)

    # Model
    p.add_argument("--d_model", type=int, default=ExpConfig.d_model)
    p.add_argument("--n_layers", type=int, default=ExpConfig.n_layers)
    p.add_argument("--n_heads", type=int, default=ExpConfig.n_heads)
    p.add_argument("--ff_dim", type=int, default=ExpConfig.ff_dim)
    p.add_argument("--dropout", type=float, default=ExpConfig.dropout)
    p.add_argument("--max_seq_len", type=int, default=ExpConfig.max_seq_len)
    p.add_argument("--no_tie_weights", action="store_true", help="Disable weight tying")

    # MoE
    p.add_argument("--moe_n_experts", type=int, default=ExpConfig.moe_n_experts)
    p.add_argument("--moe_top_k", type=int, default=ExpConfig.moe_top_k)
    p.add_argument("--moe_activation", type=str, default=ExpConfig.moe_activation)
    p.add_argument("--moe_noisy_gate", action="store_true", default=ExpConfig.moe_noisy_gate)

    # Training
    p.add_argument("--lr", type=float, default=ExpConfig.lr)
    p.add_argument("--weight_decay", type=float, default=ExpConfig.weight_decay)
    p.add_argument("--max_steps", type=int, default=ExpConfig.max_steps)
    p.add_argument("--log_every", type=int, default=ExpConfig.log_every)
    p.add_argument("--eval_every", type=int, default=ExpConfig.eval_every)
    p.add_argument("--grad_clip", type=float, default=ExpConfig.grad_clip)
    p.add_argument("--no_amp", action="store_true", help="Disable AMP even if CUDA is available")

    # Misc
    p.add_argument("--seed", type=int, default=ExpConfig.seed)
    p.add_argument("--runs_dir", type=str, default=ExpConfig.runs_dir)
    p.add_argument("--run_tag", type=str, default=ExpConfig.run_tag)

    # Control which variants to run
    p.add_argument("--run_dense", action="store_true", help="Run dense baseline")
    p.add_argument("--run_moe", action="store_true", help="Run MoE variant")

    args = p.parse_args()

    cfg = ExpConfig(
        model_name=args.model_name,
        lowercase=bool(args.lowercase),
        max_length=args.max_length,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        data_root=args.data_root,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        tie_weights=not args.no_tie_weights,
        # MoE toggles are set per-variant; defaults here are kept but overridden in run_one
        moe_enabled=False,
        moe_n_experts=args.moe_n_experts,
        moe_top_k=args.moe_top_k,
        moe_activation=args.moe_activation,
        moe_noisy_gate=bool(args.moe_noisy_gate),
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        seed=args.seed,
        runs_dir=args.runs_dir,
        run_tag=args.run_tag,
    )

    # Default to running both if none specified
    run_dense = args.run_dense or (not args.run_dense and not args.run_moe)
    run_moe = args.run_moe or (not args.run_dense and not args.run_moe)

    results: Dict[str, Any] = {
        "config_common": asdict(cfg),
        "results": {},
    }

    if run_dense:
        print("=== Running dense baseline ===")
        dense_res = run_one(cfg, "dense")
        results["results"]["dense"] = dense_res

    if run_moe:
        print("=== Running MoE variant ===")
        moe_res = run_one(cfg, "moe")
        results["results"]["moe"] = moe_res

    suffix = "dense_and_moe" if run_dense and run_moe else ("dense" if run_dense else "moe")
    save_results(results, cfg, suffix=suffix)


if __name__ == "__main__":
    main()