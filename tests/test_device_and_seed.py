from __future__ import annotations

import numpy as np
import torch

from project_chimera.utils.device import device_name, get_device
from project_chimera.utils.repro import set_seed


def test_get_device_returns_torch_device():
    dev = get_device()
    assert isinstance(dev, torch.device)


def test_device_name_str():
    name = device_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_seed_repro_numpy_and_torch():
    set_seed(1234, deterministic=True)
    a1 = np.random.rand(3)
    b1 = torch.randn(3)

    set_seed(1234, deterministic=True)
    a2 = np.random.rand(3)
    b2 = torch.randn(3)

    assert np.allclose(a1, a2)
    assert torch.allclose(b1, b2)
