from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn

from .experts import ExpertConfig, FFNExpert, ExpertParallel
from .gating import GatingConfig, TopKGating


@dataclass
class MoEConfig:
    d_model: int
    ff_dim: int
    n_experts: int
    k: int = 1
    dropout: float = 0.0
    activation: str = "gelu"
    noisy_gate: bool = False
    capacity_factor: float = 1.0  # placeholder; not enforced in this stub
    aux_loss_coef: float = 0.01   # coefficient for load-balancing loss (future use)


class MoELayer(nn.Module):
    """
    Minimal MoE layer stub:
      - Creates N FFN experts
      - Uses Top-K gating to select expert indices and weights per token
      - Performs a simplified 'soft' combine via averaging experts' outputs for now

    NOTE: This is a placeholder focusing on interface shape and API only.
          It does not implement efficient dispatch (gather/scatter) or
          capacity constraints yet.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg

        # Experts
        exp_cfg = ExpertConfig(d_model=cfg.d_model, ff_dim=cfg.ff_dim, dropout=cfg.dropout, activation=cfg.activation)
        experts: List[nn.Module] = [FFNExpert(exp_cfg) for _ in range(cfg.n_experts)]
        self.experts = ExpertParallel(experts)

        # Gating
        gate_cfg = GatingConfig(
            d_model=cfg.d_model,
            n_experts=cfg.n_experts,
            k=cfg.k,
            noisy_gate=cfg.noisy_gate,
            capacity_factor=cfg.capacity_factor,
        )
        self.gate = TopKGating(gate_cfg)

        # Final dropout (optional)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            x: [B, T, C]
        Returns:
            dict with:
              'out': [B, T, C] combined output
              'aux_loss': scalar tensor (placeholder 0.0)
              'routing': optional diagnostics (indices, weights)
        """
        B, T, C = x.shape
        device = x.device

        # Compute routing
        topk_idx, topk_weight = self.gate(x)  # [B, T, k], [B, T, k]

        # Simplified combine: run all experts on x, then select/weight outputs
        # In a full implementation, only routed tokens would be sent to selected experts
        # via gather/scatter for efficiency.
        all_exp_out = []
        for e in self.experts.experts:
            all_exp_out.append(e(x))  # each [B, T, C]
        # Stack to [E, B, T, C]
        exp_out = torch.stack(all_exp_out, dim=0)  # [E, B, T, C]

        # Build a mask/weights tensor to combine top-k expert outputs:
        # For simplicity, convert indices/weights to [E, B, T] weights
        E = exp_out.shape[0]
        weights = torch.zeros(E, B, T, device=device, dtype=exp_out.dtype)

        # Scatter top-k weights into the expert dimension
        # topk_idx/topk_weight: [B, T, k]
        # We iterate over k and add weights to corresponding expert slice
        k = self.cfg.k
        for i in range(k):
            idx_i = topk_idx[:, :, i]  # [B, T]
            w_i = topk_weight[:, :, i]  # [B, T]
            # Scatter add on expert axis
            weights = weights.index_put(
                (idx_i.unsqueeze(0), torch.arange(B, device=device).unsqueeze(0).unsqueeze(-1).expand(E, -1, T), torch.arange(T, device=device).unsqueeze(0).unsqueeze(0).expand(E, B, -1)),
                torch.zeros_like(weights),
                accumulate=False,
            )  # reset, we'll fill below to avoid mixing; we do a manual fill next

        # Manual fill since index_put with multiple advanced indices is complex; do a loop over batch/time for clarity
        weights.zero_()
        for b in range(B):
            for t in range(T):
                for i in range(k):
                    e_idx = int(topk_idx[b, t, i].item())
                    weights[e_idx, b, t] += topk_weight[b, t, i]

        # Combine: [E, B, T, C] * [E, B, T, 1] -> sum over E
        combined = (exp_out * weights.unsqueeze(-1)).sum(dim=0)  # [B, T, C]
        combined = self.drop(combined)

        aux_loss = torch.zeros((), device=device)  # placeholder for load-balancing loss

        return {
            "out": combined,
            "aux_loss": aux_loss,
            "routing": {
                "topk_idx": topk_idx,
                "topk_weight": topk_weight,
            },
        }


class MoEFFNWrapper(nn.Module):
    """
    Drop-in replacement for a dense FFN inside a Transformer block.
    Matches the dense FFN signature: forward(x) -> Tensor
    Returns only the transformed tensor; aux losses can be handled by the caller if desired.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.moe = MoELayer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.moe(x)
        return out["out"]