"""
src/__init__.py
---------------
Public API surface for the lyapunov-edge-inference package.

Domain model dataclasses
------------------------
    Frame, Detection, ControllerState, ControllerAction,
    Transition, TelemetryRecord

Controller & Agent
------------------
    AdaptiveController, LyapunovPPOAgent

Inference engines
-----------------
    DetectionEngine, SegmentationEngine

Safety layers
-------------
    ConformalPredictor, LyapunovCritic, LyapunovManager, TransitionModel

Baselines
---------
    BaseController, FixedHighQualityController, FixedLowLatencyController,
    RuleBasedController, PIDController, CONTROLLER_REGISTRY

Monitoring & telemetry
----------------------
    MetricsWindow, GPUMonitor, TelemetryLogger,
    DriftMonitor, FrameTimer

Infrastructure
--------------
    CameraCapture, Preprocessor, LatencyEnv, LatencyPredictor

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

# Controller & agent — lazy imports to avoid hard GPU dependency at package level.
# These are re-exported so that ``from src import AdaptiveController`` works when
# the runtime environment has the required backends installed.
try:
    from src.controller import AdaptiveController
except ImportError:
    pass

try:
    from src.agent_lyapunov_ppo import LyapunovPPOAgent
except ImportError:
    pass

# Inference engines — guarded because they require TensorRT + PyCUDA.
try:
    from src.detection import DetectionEngine
except ImportError:
    pass

try:
    from src.segmentation import SegmentationEngine
except ImportError:
    pass

# Safety layers
try:
    from src.conformal import ConformalPredictor
except ImportError:
    pass

try:
    from src.lyapunov import LyapunovCritic, LyapunovManager, TransitionModel
except ImportError:
    pass

try:
    from src.lagrangian import LagrangianDual
except ImportError:
    pass

# Baselines
try:
    from src.baselines import (
        CONTROLLER_REGISTRY,
        BaseController,
        FixedHighQualityController,
        FixedLowLatencyController,
        PIDController,
        RuleBasedController,
    )
except ImportError:
    pass

# Monitoring & telemetry
try:
    from src.monitoring import GPUMonitor, MetricsWindow, TelemetryLogger
except ImportError:
    pass

try:
    from src.drift import DriftMonitor
except ImportError:
    pass

try:
    from src.telemetry import FrameTimer
except ImportError:
    pass

# Infrastructure
try:
    from src.camera import CameraCapture
except ImportError:
    pass

try:
    from src.preprocess import Preprocessor
except ImportError:
    pass

try:
    from src.env import LatencyEnv
except ImportError:
    pass

try:
    from src.latency_predictor import LatencyPredictor
except ImportError:
    pass

__all__ = [
    # ── Domain model ──────────────────────────────────────────────────────────
    "Frame",
    "Detection",
    "ControllerState",
    "ControllerAction",
    "Transition",
    "TelemetryRecord",
    # ── Controller & agent ────────────────────────────────────────────────────
    "AdaptiveController",
    "LyapunovPPOAgent",
    # ── Inference engines ─────────────────────────────────────────────────────
    "DetectionEngine",
    "SegmentationEngine",
    # ── Safety layers ─────────────────────────────────────────────────────────
    "ConformalPredictor",
    "LyapunovCritic",
    "LyapunovManager",
    "TransitionModel",
    "LagrangianDual",
    # ── Baselines ─────────────────────────────────────────────────────────────
    "BaseController",
    "FixedHighQualityController",
    "FixedLowLatencyController",
    "RuleBasedController",
    "PIDController",
    "CONTROLLER_REGISTRY",
    # ── Monitoring & telemetry ────────────────────────────────────────────────
    "MetricsWindow",
    "GPUMonitor",
    "TelemetryLogger",
    "DriftMonitor",
    "FrameTimer",
    # ── Infrastructure ────────────────────────────────────────────────────────
    "CameraCapture",
    "Preprocessor",
    "LatencyEnv",
    "LatencyPredictor",
    # ── Utilities ─────────────────────────────────────────────────────────────
    "load_config",
    "load_all_configs",
    "setup_device",
    "setup_logging",
    "resolve_checkpoint",
    "compute_sha256",
    "verify_checkpoint",
]
