"""
src/__init__.py
---------------
Public API surface for the lyapunov-edge-inference package.

Domain model dataclasses
------------------------
    Frame, Detection, ControllerState, ControllerAction,
    Transition, TelemetryRecord

Utility functions
-----------------
    load_config, load_all_configs,
    setup_device, setup_logging,
    resolve_checkpoint, compute_sha256, verify_checkpoint
"""

from src.state_features import (
    ControllerAction,
    ControllerState,
    Detection,
    Frame,
    TelemetryRecord,
    Transition,
)
from src.utils import (
    compute_sha256,
    load_all_configs,
    load_config,
    resolve_checkpoint,
    setup_device,
    setup_logging,
    verify_checkpoint,
)

__all__ = [
    # ── Domain model ──────────────────────────────────────────────────────────
    "Frame",
    "Detection",
    "ControllerState",
    "ControllerAction",
    "Transition",
    "TelemetryRecord",
    # ── Utilities ─────────────────────────────────────────────────────────────
    "load_config",
    "load_all_configs",
    "setup_device",
    "setup_logging",
    "resolve_checkpoint",
    "compute_sha256",
    "verify_checkpoint",
]
