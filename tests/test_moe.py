import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.project_chimera.moe.layer import MoEConfig, MoELayer
from src.project_chimera.utils.repro import set_seed
from src.project_chimera.utils.device import get_device


def make_moe(d_model=16, ff_dim=32, n_experts=4, k=2, capacity_factor=1.0, dropout=0.0, activation="gelu", noisy_gate=False, aux_loss_coef=0.01):
    cfg = MoEConfig(
        d_model=d_model,
        ff_dim=ff_dim,
        n_experts=n_experts,
        k=k,
        dropout=dropout,
        activation=activation,
        noisy_gate=noisy_gate,
        capacity_factor=capacity_factor,
        aux_loss_coef=aux_loss_coef,
    )
    return MoELayer(cfg)


def test_output_shape_and_finite_loss():
    device = get_device(prefer_cuda=False, prefer_mps=False)  # keep CPU deterministic for CI
    set_seed(123, deterministic=True)
    B, T, C = 3, 5, 16
    moe = make_moe(d_model=C, ff_dim=32, n_experts=4, k=2, capacity_factor=1.0, dropout=0.0).to(device)
    x = torch.randn(B, T, C, device=device)
    out = moe(x)
    y = out["out"]
    aux = out["aux_loss"]
    assert y.shape == (B, T, C)
    assert torch.isfinite(y).all().item() is True
    assert torch.isfinite(aux).all().item() is True
    assert aux.dim() == 0  # scalar


def test_routing_indices_and_capacity_respected():
    device = get_device(prefer_cuda=False, prefer_mps=False)
    set_seed(321, deterministic=True)
    B, T, C = 4, 7, 12
    E, K = 3, 2
    # choose small capacity_factor to force drops
    cap_factor = 0.25
    moe = make_moe(d_model=C, ff_dim=24, n_experts=E, k=K, capacity_factor=cap_factor, dropout=0.0).to(device)
    x = torch.randn(B, T, C, device=device)

    out = moe(x)
    routing = out["routing"]
    topk_idx = routing["topk_idx"]  # [B,T,K]
    topk_weight = routing["topk_weight"]
    capacity = routing["capacity"]
    expert_load = routing["expert_load"]  # [E]

    # Indices are in range
    assert topk_idx.min().item() >= 0
    assert topk_idx.max().item() < E

    # Weights sum to approximately 1 across K per token (softmax of selected)
    wsum = topk_weight.sum(dim=-1)
    assert torch.allclose(wsum, torch.ones_like(wsum), atol=1e-5)

    # Capacity computation matches layer formula
    N = B * T
    expected_capacity = int((cap_factor * N * max(1, K)) / E + 0.9999)
    expected_capacity = max(1, expected_capacity)
    assert capacity == expected_capacity

    # Each expert's effective assigned tokens (counted) must be <= capacity
    # expert_load is incremented by 1 per accepted token-choice
    assert (expert_load <= capacity + 1e-5).all().item() is True


def test_determinism_fixed_seed():
    device = get_device(prefer_cuda=False, prefer_mps=False)
    set_seed(777, deterministic=True)
    B, T, C = 2, 6, 8
    E, K = 4, 2

    moe1 = make_moe(d_model=C, ff_dim=16, n_experts=E, k=K, noisy_gate=False).to(device)
    x = torch.randn(B, T, C, device=device)
    # Copy weights to a second instance for a fair determinism check
    moe2 = make_moe(d_model=C, ff_dim=16, n_experts=E, k=K, noisy_gate=False).to(device)
    moe2.load_state_dict(moe1.state_dict())

    # With same seed and inputs, outputs should match exactly on CPU
    out1 = moe1(x)
    set_seed(777, deterministic=True)  # reset just in case
    out2 = moe2(x)

    assert torch.allclose(out1["out"], out2["out"], atol=0.0)
    assert torch.allclose(out1["aux_loss"], out2["aux_loss"], atol=0.0)
    # Routing determinism
    assert torch.equal(out1["routing"]["topk_idx"], out2["routing"]["topk_idx"])
    assert torch.allclose(out1["routing"]["topk_weight"], out2["routing"]["topk_weight"], atol=0.0)


@pytest.mark.parametrize("K", [1, 2, 3])
def test_topk_distribution_sanity(K):
    device = get_device(prefer_cuda=False, prefer_mps=False)
    set_seed(999, deterministic=True)
    B, T, C = 8, 10, 16
    E = 6
    moe = make_moe(d_model=C, ff_dim=32, n_experts=E, k=K, dropout=0.0, noisy_gate=False, capacity_factor=1.0).to(device)
    x = torch.randn(B, T, C, device=device)

    out = moe(x)
    topk_idx = out["routing"]["topk_idx"]  # [B,T,K]

    # Each token should have K distinct experts unless ties cause identical indices from topk over logits.
    # Since topk over logits can produce equal indices only if E &lt; K which we avoid, we allow duplicates but
    # expect the unique count to be close to K for most tokens.
    uniq_counts = torch.tensor([topk_idx[b, t].unique().numel() for b in range(B) for t in range(T)], device=topk_idx.device)
    # Expect the number of unique experts per token to be close to K (allow tiny slack for rare ties)\nassert uniq_counts.float().mean().item() >= max(1.0, K - 0.25)\n\n# Expert utilization should be non-trivial across many experts\nflat_idx = topk_idx.reshape(-1)\n# Count how many times each expert is selected\ncounts = torch.bincount(flat_idx, minlength=E).float()\n# At least half of experts should be selected at least once in this sample\nassert (counts > 0).sum().item() >= max(1, E // 2)


def test_backward_and_finite_gradients():
    device = get_device(prefer_cuda=False, prefer_mps=False)
    set_seed(2024, deterministic=True)
    B, T, C = 3, 4, 12
    E, K = 4, 2
    moe = make_moe(d_model=C, ff_dim=24, n_experts=E, k=K, capacity_factor=1.0, dropout=0.0).to(device)
    x = torch.randn(B, T, C, device=device, requires_grad=True)

    out = moe(x)
    y = out["out"]
    aux = out["aux_loss"]
    # simple scalar loss: MSE to zero plus aux
    loss = (y.pow(2).mean()) + aux
    loss.backward()

    # Gradients exist and are finite for inputs and some parameters
    assert torch.isfinite(x.grad).all().item() is True
    any_param_has_grad = any(p.grad is not None and torch.isfinite(p.grad).all().item() for p in moe.parameters())
    assert any_param_has_grad is True