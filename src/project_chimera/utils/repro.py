from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set seeds across Python, NumPy, and PyTorch for reproducible experiments.

    Args:
        seed: Seed value to set across random generators.
        deterministic: If True, enable deterministic flags (may impact performance).
    """
    # Python stdlib and numpy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN determinism settings
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """
    Worker init function for DataLoader workers to make them deterministic.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
