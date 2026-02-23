"""
Device selection utilities for Part 4 training/evaluation scripts.
"""

from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> str:
    """
    Resolve runtime device with robust fallbacks.

    Priority when requested is None/"auto":
      1) CUDA
      2) Apple MPS
      3) CPU

    Special handling for Apple users:
      If requested == "cuda" but CUDA is unavailable and MPS is available,
      this returns "mps".
    """
    if requested is None:
        requested = "auto"
    req = requested.lower()

    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if req in {"auto", "best"}:
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    if req == "cuda":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    if req in {"mps", "metal", "apple"}:
        if mps_ok:
            return "mps"
        raise RuntimeError(
            "Requested Apple GPU (mps), but MPS is unavailable in this runtime. "
            "Please run on a macOS environment with MPS-enabled PyTorch."
        )

    if req == "cpu":
        return "cpu"

    raise ValueError(f"Unsupported device request: {requested}")
