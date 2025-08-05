#!/usr/bin/env python3
"""
Utility script to scaffold the base project structure for Project Chimera.

Creates:
- src/project_chimera/__init__.py
- src/project_chimera/utils/{repro.py,device.py}
- tests/test_device_and_seed.py
- notebooks/.gitkeep
- scripts/.gitkeep (kept except this file)
- data/README.md
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def touch(path: Path, content: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content)

def main():
    # Package init
    touch(ROOT / "src" / "project_chimera" / "__init__.py", content="# Project Chimera package\n")

    # Utils: reproducibility
    repro_content = """from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    \"""
    Set seeds across libraries for reproducible experiments.

    Args:
        seed: Seed value to set across random generators.
        deterministic: If True, enable deterministic flags (may impact performance).
    \"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For cudnn determinism (may reduce performance)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    \"""
    Worker init function for DataLoader workers to make them deterministic.
    \"""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
"""
    touch(ROOT / "src" / "project_chimera" / "utils" / "repro.py", content=repro_content)

    # Utils: device
    device_content = """from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    \"""
    Select the appropriate torch device.

    Order of preference:
    - CUDA if available and prefer_cuda is True
    - MPS (Apple Silicon) if available and prefer_mps is True
    - CPU otherwise
    \"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    # macOS MPS backend
    if prefer_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device | None = None) -> str:
    \"""
    Human-friendly device name.
    \"""
    d = device or get_device()
    if d.type == "cuda":
        idx = d.index or 0
        return f"cuda:{idx} - " + torch.cuda.get_device_name(idx)
    if d.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
"""
    touch(ROOT / "src" / "project_chimera" / "utils" / "device.py", content=device_content)

    # Tests
    test_content = """from __future__ import annotations

import numpy as np
import torch

from project_chimera.utils.device import get_device
from project_chimera.utils.repro import set_seed


def test_get_device_returns_torch_device():
    dev = get_device()
    assert isinstance(dev, torch.device)


def test_seed_repro_numpy_and_torch():
    set_seed(1234, deterministic=True)
    a1 = np.random.rand(3)
    b1 = torch.randn(3)

    set_seed(1234, deterministic=True)
    a2 = np.random.rand(3)
    b2 = torch.randn(3)

    assert np.allclose(a1, a2)
    assert torch.allclose(b1, b2)
"""
    touch(ROOT / "tests" / "test_device_and_seed.py", content=test_content)

    # Notebooks placeholder
    touch(ROOT / "notebooks" / ".gitkeep", content="")

    # Scripts placeholder (besides this file)
    touch(ROOT / "scripts" / ".gitkeep", content="")

    # Data readme
    data_readme = """# Data Directory

- raw/: place raw downloads here (ignored by git)
- cache/: processed/cache artifacts (ignored by git)
- tmp/: temporary files (ignored by git)

This folder is intentionally kept out of version control (see .gitignore) except this README.
"""
    touch(ROOT / "data" / "README.md", content=data_readme)

    print("Project structure created.")

if __name__ == "__main__":
    main()