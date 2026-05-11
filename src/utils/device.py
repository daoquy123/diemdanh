"""Device selection helper that gracefully falls back to CPU."""
from __future__ import annotations

import warnings

import torch


def select_device(preferred: str = "cuda") -> torch.device:
    """Return the requested torch device, falling back to CPU when unavailable."""
    preferred = (preferred or "").lower()
    if preferred.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(preferred if ":" in preferred else "cuda:0")
        warnings.warn(
            "Requested CUDA but torch.cuda.is_available() is False — using CPU. "
            "Install a GPU build of PyTorch (see https://pytorch.org/get-started/locally/) "
            "and ensure NVIDIA drivers are installed.",
            UserWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
