from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    """
    Select the appropriate torch device.

    Preference order:
    1) CUDA if available and prefer_cuda is True
    2) MPS (Apple Silicon) if available and prefer_mps is True
    3) CPU
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device | None = None) -> str:
    """
    Human-friendly device name for logging.
    """
    d = device or get_device()
    if d.type == "cuda":
        idx = d.index or 0
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "NVIDIA GPU"
        return f"cuda:{idx} - {name}"
    if d.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
