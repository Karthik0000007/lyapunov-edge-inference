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

import logging as _logging

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

_logger = _logging.getLogger(__name__)

# Controller & agent — lazy imports to avoid hard GPU dependency at package level.
# These are re-exported so that ``from src import AdaptiveController`` works when
# the runtime environment has the required backends installed.
try:
    from src.controller import AdaptiveController
except ImportError as e:
    _logger.debug("AdaptiveController not available: %s", e)

try:
    from src.agent_lyapunov_ppo import LyapunovPPOAgent
except ImportError as e:
    _logger.debug("LyapunovPPOAgent not available: %s", e)

# Inference engines — guarded because they require TensorRT + PyCUDA.
try:
    from src.detection import DetectionEngine
except ImportError as e:
    _logger.debug("DetectionEngine not available (TensorRT required): %s", e)

try:
    from src.segmentation import SegmentationEngine
except ImportError as e:
    _logger.debug("SegmentationEngine not available (TensorRT required): %s", e)

# Safety layers
try:
    from src.conformal import ConformalPredictor
except ImportError as e:
    _logger.debug("ConformalPredictor not available: %s", e)

try:
    from src.lyapunov import LyapunovCritic, LyapunovManager, TransitionModel
except ImportError as e:
    _logger.debug("Lyapunov components not available: %s", e)

try:
    from src.lagrangian import LagrangianDual
except ImportError as e:
    _logger.debug("LagrangianDual not available: %s", e)

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
except ImportError as e:
    _logger.debug("Baselines not available: %s", e)

# Monitoring & telemetry
try:
    from src.monitoring import GPUMonitor, MetricsWindow, TelemetryLogger
except ImportError as e:
    _logger.debug("Monitoring components not available: %s", e)

try:
    from src.drift import DriftMonitor
except ImportError as e:
    _logger.debug("DriftMonitor not available: %s", e)

try:
    from src.telemetry import FrameTimer
except ImportError as e:
    _logger.debug("FrameTimer not available: %s", e)

# Infrastructure
try:
    from src.camera import CameraCapture
except ImportError as e:
    _logger.debug("CameraCapture not available: %s", e)

try:
    from src.preprocess import Preprocessor
except ImportError as e:
    _logger.debug("Preprocessor not available: %s", e)

try:
    from src.env import LatencyEnv
except ImportError as e:
    _logger.debug("LatencyEnv not available: %s", e)

try:
    from src.latency_predictor import LatencyPredictor
except ImportError as e:
    _logger.debug("LatencyPredictor not available: %s", e)

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
