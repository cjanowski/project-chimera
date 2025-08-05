from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GatingConfig:
    d_model: int
    n_experts: int
    k: int = 1
    noisy_gate: bool = False
    capacity_factor: float = 1.0  # placeholder for future token capacity


class TopKGating(nn.Module):
    """
    Simple top-k gating producing dispatch (one-hot or sparse) indices and expert weights.
    This module does not perform token movement; it only computes routing decisions.
    """

    def __init__(self, cfg: GatingConfig):
        super().__init__()
        assert 1 <= cfg.k <= cfg.n_experts, "k must be in [1, n_experts]"
        self.cfg = cfg
        self.w_gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C] hidden states

        Returns:
            topk_idx: [B, T, k] indices of experts for each token
            topk_weight: [B, T, k] normalized weights per selected expert
        """
        B, T, C = x.shape
        gate_logits = self.w_gate(x)  # [B, T, E]
        if self.cfg.noisy_gate and self.training:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * 1e-2

        topk_weight, topk_idx = torch.topk(gate_logits, k=self.cfg.k, dim=-1)  # [B, T, k]
        topk_weight = F.softmax(topk_weight, dim=-1)  # normalize among selected experts
        return topk_idx, topk_weight