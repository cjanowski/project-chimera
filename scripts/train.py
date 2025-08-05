#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn

from project_chimera.utils.device import device_name, get_device
from project_chimera.utils.repro import set_seed


@dataclass
class TrainConfig:
    seed: int = 42
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.1
    vocab_size: int = 256
    seq_len: int = 64
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 20


class TinyDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        blocks = [TinyDecoderBlock(cfg.d_model, cfg.n_heads, cfg.ff_dim, cfg.dropout) for _ in range(cfg.n_layers)]
        self.blocks = nn.Sequential(*blocks)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        device = idx.device
        pos = torch.arange(0, t, device=device).unsqueeze(0).expand(b, t)
        x = self.tok(idx) + self.pos(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def synthetic_batch(cfg: TrainConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
    return x, y


def train_step(model: nn.Module, opt: torch.optim.Optimizer, loss_fn: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    opt.step()
    return float(loss.item())


def main():
    parser = argparse.ArgumentParser(description="Tiny training smoke test")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    cfg = TrainConfig(steps=args.steps, seed=args.seed)

    set_seed(cfg.seed, deterministic=True)
    device = get_device()
    print(f"Using device: {device} ({device_name(device)})")

    model = TinyDecoderLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(cfg.steps):
        x, y = synthetic_batch(cfg, device)
        loss = train_step(model, opt, loss_fn, x, y)
        if (step + 1) % 5 == 0 or step == 0:
            print(f"step {step+1}/{cfg.steps} - loss {loss:.4f}")

    print("Training smoke test finished.")

if __name__ == "__main__":
    main()