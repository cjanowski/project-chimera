from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class ExpertConfig:
    d_model: int
    ff_dim: int
    dropout: float = 0.0
    activation: str = "gelu"


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class FFNExpert(nn.Module):
    """
    Standard Transformer FFN expert: Linear - Act - Dropout - Linear - Dropout
    """

    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.ff_dim)
        self.act = _get_activation(cfg.activation)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(cfg.ff_dim, cfg.d_model)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ExpertParallel(nn.Module):
    """
    Holds a list of experts with identical shapes. This module does not perform routing;
    it simply provides access to the experts list for external dispatchers.
    """

    def __init__(self, experts: List[nn.Module]):
        super().__init__()
        self.experts = nn.ModuleList(experts)

    def __len__(self) -> int:
        return len(self.experts)

    def forward(self, *args, **kwargs):
        raise RuntimeError("ExpertParallel does not implement forward. Use routing layer to call experts.")