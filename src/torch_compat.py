"""
torch_compat.py
───────────────
Compatibility utilities for PyTorch version differences.

Provides helpers that work across different PyTorch versions, particularly
for features like weights_only in torch.load() which were introduced in
newer versions (2.13+).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def torch_load_compat(
    path: Path | str,
    map_location: Optional[str | torch.device] = None,
) -> Any:
    """Load a checkpoint with torch.load, supporting older/newer torch signatures.

    Attempts to call torch.load(..., weights_only=True) first for security in
    newer PyTorch versions (2.13+). If that raises a TypeError because the
    installed torch doesn't recognize weights_only, falls back to calling
    torch.load(...) without the parameter.

    This ensures compatibility across torch versions while maintaining security
    when possible.

    Parameters
    ----------
    path : Path | str
        Path to the checkpoint file to load.
    map_location : str | torch.device, optional
        Device mapping (e.g., "cpu", "cuda:0") for relocating tensors.

    Returns
    -------
    Any
        Loaded checkpoint state dict or object.
    """
    path = str(path)  # Ensure string for torch.load
    kwargs: dict[str, Any] = {}
    if map_location is not None:
        kwargs["map_location"] = map_location

    try:
        # Try with weights_only=True first (preferred in newer torch versions)
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError as e:
        # If weights_only is not recognized, fall back to standard call
        if "weights_only" in str(e):
            return torch.load(path, **kwargs)
        # Re-raise if it's a different TypeError
        raise
