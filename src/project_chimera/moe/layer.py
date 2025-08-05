from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    capacity_factor: float = 1.0
    aux_loss_coef: float = 0.01


def _compute_load_balancing_loss(gate_probs: torch.Tensor, expert_assignments: torch.Tensor, n_experts: int) -> torch.Tensor:
    """
    Load balancing loss similar to Switch Transformer:
      - gate_probs: [B, T, E] softmax over experts per token (pre-topk)
      - expert_assignments: [E] total tokens (or mass) per expert
    Encourages uniform utilization by minimizing the squared coefficient of variation between expected and actual loads.
    """
    # Expected fraction per expert from average gate probability
    frac_expected = gate_probs.mean(dim=(0, 1))  # [E]
    frac_expected = frac_expected / (frac_expected.sum() + 1e-9)

    # Actual fraction per expert from assignments count/mass
    total = expert_assignments.sum() + 1e-9
    frac_actual = expert_assignments / total

    # Squared difference loss
    loss = ((frac_expected - frac_actual).pow(2)).mean()
    return loss


class MoELayer(nn.Module):
    """
    MoE layer with efficient token dispatch:
      - Top-K gating produces expert indices and weights
      - Capacity controls max tokens per expert; overflow tokens are dropped (or routed to first selected expert until capacity)
      - Tokens are gathered per-expert, processed, then scattered back and combined with weights
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg

        # Experts
        exp_cfg = ExpertConfig(d_model=cfg.d_model, ff_dim=cfg.ff_dim, dropout=cfg.dropout, activation=cfg.activation)
        self.experts = ExpertParallel([FFNExpert(exp_cfg) for _ in range(cfg.n_experts)])

        # Gating
        gate_cfg = GatingConfig(
            d_model=cfg.d_model,
            n_experts=cfg.n_experts,
            k=cfg.k,
            noisy_gate=cfg.noisy_gate,
            capacity_factor=cfg.capacity_factor,
        )
        self.gate = TopKGating(gate_cfg)

        # Dropout
        self.drop = nn.Dropout(cfg.dropout)

        # A small linear to obtain full gate probs for aux loss; reuse gating weights
        self._gate_linear = self.gate.w_gate  # nn.Linear(d_model, E, bias=False)

    def _capacity(self, num_tokens: int) -> int:
        # capacity per expert = ceil(capacity_factor * num_tokens * k / n_experts)
        capacity = int((self.cfg.capacity_factor * num_tokens * max(1, self.cfg.k)) / self.cfg.n_experts + 0.9999)
        return max(1, capacity)

    def _compute_gate_probs(self, x: torch.Tensor) -> torch.Tensor:
        # gate logits -> softmax over experts for aux loss signal
        logits = self._gate_linear(x)  # [B, T, E]
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            x: [B, T, C]
        Returns:
            dict(out=[B, T, C], aux_loss=scalar, routing=diagnostics)
        """
        B, T, C = x.shape
        device = x.device
        E = self.cfg.n_experts
        K = self.cfg.k
        N = B * T

        # Flatten tokens to [N, C] for routing convenience
        x_flat = x.reshape(N, C)

        # Compute top-k routing on [B, T, C] for readability
        topk_idx, topk_weight = self.gate(x)  # [B, T, K], [B, T, K]
        topk_idx_flat = topk_idx.reshape(N, K)
        topk_weight_flat = topk_weight.reshape(N, K)

        # Capacity per expert
        capacity = self._capacity(N)

        # Build per-expert token lists (indices into x_flat), respecting capacity
        expert_token_indices: List[List[int]] = [[] for _ in range(E)]
        expert_token_weights: List[List[float]] = [[] for _ in range(E)]
        token_positions_in_expert: List[List[int]] = [[] for _ in range(E)]  # position for scatter back

        # Track how many tokens each expert gets
        expert_load = torch.zeros(E, device=device, dtype=torch.float32)

        # For each token, iterate its K choices; assign to each chosen expert until capacity
        for n in range(N):
            for k_i in range(K):
                e_idx = int(topk_idx_flat[n, k_i].item())
                if len(expert_token_indices[e_idx]) < capacity:
                    expert_token_indices[e_idx].append(n)
                    expert_token_weights[e_idx].append(float(topk_weight_flat[n, k_i].item()))
                    token_positions_in_expert[e_idx].append(len(expert_token_indices[e_idx]) - 1)
                    expert_load[e_idx] += 1.0
                # If capacity filled, we simply drop overflow for that expert-choice

        # Prepare output tensor
        out = torch.zeros(N, C, device=device, dtype=x.dtype)

        # Dispatch to each expert, process, and combine weighted outputs
        for e_idx, expert in enumerate(self.experts.experts):
            idxs = expert_token_indices[e_idx]
            if len(idxs) == 0:
                continue
            idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)  # [M_e]
            # Gather expert inputs
            x_e = x_flat.index_select(0, idx_tensor)  # [M_e, C]
            # Expert expects [B,T,C]-shaped, but is FFN over last dim; we can keep [M_e, 1, C] -> [M_e, 1, C]
            x_e = x_e.unsqueeze(1)
            y_e = expert(x_e)  # [M_e, 1, C]
            y_e = y_e.squeeze(1)  # [M_e, C]
            # Weights for these tokens
            w_e = torch.tensor(expert_token_weights[e_idx], device=device, dtype=y_e.dtype).unsqueeze(-1)  # [M_e, 1]
            # Scatter-add back to output with weighting
            out.index_add_(0, idx_tensor, y_e * w_e)

        out = out.reshape(B, T, C)
        out = self.drop(out)

        # Aux load-balancing loss using expected probs vs actual loads
        gate_probs = self._compute_gate_probs(x)  # [B, T, E]
        lb_loss = _compute_load_balancing_loss(gate_probs, expert_load.detach(), E)
        aux_loss = self.cfg.aux_loss_coef * lb_loss

        return {
            "out": out,
            "aux_loss": aux_loss,
            "routing": {
                "topk_idx": topk_idx,
                "topk_weight": topk_weight,
                "capacity": capacity,
                "expert_load": expert_load,
            },
        }


class MoEFFNWrapper(nn.Module):
    """
    Drop-in replacement for a dense FFN inside a Transformer block.
    Returns only the transformed tensor; aux losses can be handled by the caller if desired.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.moe = MoELayer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.moe(x)
        return out["out"]