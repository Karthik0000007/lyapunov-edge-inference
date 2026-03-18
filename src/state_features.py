"""
src/state_features.py
---------------------
Domain model dataclasses and state normalization for the Lyapunov-constrained
edge inference pipeline.

Public API
----------
    Frame, Detection, ControllerState, ControllerAction,
    Transition, TelemetryRecord
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

# ── Normalization constants ───────────────────────────────────────────────────
# Fixed ranges used by ControllerState.to_tensor() to map each feature to [0, 1].
# These defaults are tuned for RTX 3050 + NEU-DET dataset. For other hardware
# or datasets, configure via config/pipeline.yaml → normalization section.

_LATENCY_MIN_MS: float = 5.0       # Minimum expected frame latency (ms)
_LATENCY_MAX_MS: float = 100.0     # Maximum expected frame latency (ms)

_DETECTION_COUNT_MIN: float = 0.0   # Minimum detections (always 0)
_DETECTION_COUNT_MAX: float = 50.0  # Max expected detections per frame (dataset-specific)

_GPU_UTIL_MIN: float = 0.0          # GPU utilization floor (%)
_GPU_UTIL_MAX: float = 100.0        # GPU utilization ceiling (%)

_GPU_TEMP_MIN: float = 30.0         # Idle GPU temperature (°C, hardware-specific)
_GPU_TEMP_MAX: float = 100.0        # Thermal throttle threshold (°C, hardware-specific)

_NUM_ACTIONS: int = 18              # 3 resolutions × 3 thresholds × 2 segmentation


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _normalize(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return _clamp01((v - lo) / (hi - lo))


# ── Detection ─────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single bounding-box detection emitted by Stage 1 (YOLOv8)."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalised [0, 1]
    class_id: int                             # Defect class index
    confidence: float                         # Detection confidence [0, 1]
    class_name: str                           # Human-readable label


# ── Frame ─────────────────────────────────────────────────────────────────────

@dataclass
class Frame:
    """Complete per-frame record produced by the inference pipeline."""

    # Sequence / timing
    frame_id: int
    timestamp_acquired: float        # time.perf_counter_ns() at camera read
    timestamp_preprocessed: float    # After CLAHE + tiling + ROI
    timestamp_detected: float        # After Stage 1 (YOLOv8)
    timestamp_segmented: float       # After Stage 2 (UNet), or None if skipped
    timestamp_completed: float       # After post-processing + output

    # Image data
    raw_image: np.ndarray            # Original BGR frame (H, W, 3)
    preprocessed: np.ndarray         # Enhanced frame at active resolution
    active_resolution: Tuple[int, int]  # (H, W) as set by controller

    # Inference outputs
    detections: List[Detection]      # Stage 1 outputs
    masks: Optional[List[np.ndarray]]  # Stage 2 outputs (None if skipped)

    # Controller metadata
    controller_action: int           # Action index 0–17 applied to this frame
    latency_ms: float                # End-to-end latency (completed − acquired)
    stage2_executed: bool            # Whether segmentation ran


# ── ControllerState ───────────────────────────────────────────────────────────

@dataclass
class ControllerState:
    """
    11-dimensional state vector fed to the RL controller.

    Feature layout
    ──────────────
    Indices 0-2  : latency features     (last, mean, P99) in ms
    Indices 3-5  : defect context       (count, mean confidence, area ratio)
    Indices 6-8  : pipeline config      (resolution_index, threshold_index, seg)
    Indices 9-10 : system resources     (GPU util, GPU temperature)
    """

    # Latency features (3)
    last_latency_ms: float     # ℓ_{t-1}: most-recent frame latency
    mean_latency_ms: float     # ℓ̄_W: windowed mean  (W = 50 frames)
    p99_latency_ms: float      # ℓ̂^(99)_W: windowed P99 estimate

    # Defect context features (3)
    detection_count: int       # n_t: detections in last frame
    mean_confidence: float     # c̄_t: mean detection confidence [0, 1]
    defect_area_ratio: float   # A_t^def: total defect area / frame area [0, 1]

    # Current pipeline configuration (3)
    resolution_index: int      # r_t: {0: 320, 1: 480, 2: 640}
    threshold_index: int       # b_t: {0: base, 1: +0.1, 2: +0.2}
    segmentation_enabled: int  # seg_t: {0: off, 1: on}

    # System features (2)
    gpu_utilization: float     # u_t^GPU: GPU compute utilisation [0, 100] %
    gpu_temperature_norm: float  # T_t^GPU: GPU temperature in °C [30, 100]

    def to_tensor(self) -> torch.Tensor:
        """Normalise all 11 features to [0, 1] and return as a (11,) float32 tensor."""
        features: List[float] = [
            # --- Latency ---
            _normalize(self.last_latency_ms,  _LATENCY_MIN_MS, _LATENCY_MAX_MS),
            _normalize(self.mean_latency_ms,  _LATENCY_MIN_MS, _LATENCY_MAX_MS),
            _normalize(self.p99_latency_ms,   _LATENCY_MIN_MS, _LATENCY_MAX_MS),
            # --- Defect context ---
            _normalize(float(self.detection_count), _DETECTION_COUNT_MIN, _DETECTION_COUNT_MAX),
            _clamp01(self.mean_confidence),
            _clamp01(self.defect_area_ratio),
            # --- Pipeline config (discrete → [0.0, 0.5, 1.0]) ---
            _clamp01(self.resolution_index  / 2.0),
            _clamp01(self.threshold_index   / 2.0),
            float(int(bool(self.segmentation_enabled))),
            # --- System ---
            _normalize(self.gpu_utilization,        _GPU_UTIL_MIN, _GPU_UTIL_MAX),
            _normalize(self.gpu_temperature_norm,   _GPU_TEMP_MIN, _GPU_TEMP_MAX),
        ]
        assert len(features) == 11, "ControllerState must have exactly 11 features"
        return torch.tensor(features, dtype=torch.float32)


# ── ControllerAction ──────────────────────────────────────────────────────────

@dataclass
class ControllerAction:
    """
    Factored action: a_t = (a^res, a^thr, a^seg).

    Flat index encoding (3 × 3 × 2 = 18 actions)
    ──────────────────────────────────────────────
        seg   = index % 2                     → {0, 1}
        thr   = (index // 2) % 3 − 1         → {-1, 0, +1}
        res   = (index // 6) % 3 − 1         → {-1, 0, +1}
    """

    action_index: int          # Flat index in [0, 17]
    resolution_delta: int      # {-1, 0, +1}: decrease / keep / increase resolution
    threshold_delta: int       # {-1, 0, +1}: tighten / keep / relax threshold
    segmentation_enabled: bool  # True = enable Stage 2, False = disable

    @staticmethod
    def from_index(index: int) -> ControllerAction:
        """Decode a flat action index into its factored components."""
        if not (0 <= index < _NUM_ACTIONS):
            raise ValueError(f"Action index must be in [0, {_NUM_ACTIONS - 1}], got {index}.")
        seg = index % 2
        thr = (index // 2) % 3 - 1   # Maps {0, 1, 2} → {-1, 0, +1}
        res = (index // 6) % 3 - 1   # Maps {0, 1, 2} → {-1, 0, +1}
        return ControllerAction(
            action_index=index,
            resolution_delta=res,
            threshold_delta=thr,
            segmentation_enabled=bool(seg),
        )

    def to_index(self) -> int:
        """Re-encode factored components back to the flat action index."""
        res_code = self.resolution_delta + 1   # {-1, 0, +1} → {0, 1, 2}
        thr_code = self.threshold_delta  + 1   # {-1, 0, +1} → {0, 1, 2}
        seg_code = int(self.segmentation_enabled)
        return res_code * 6 + thr_code * 2 + seg_code


# ── Transition ────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    """One step of experience stored in the replay / rollout buffer."""

    state: np.ndarray          # s_t  ∈ ℝ¹¹  (normalised)
    action: int                # a_t  ∈ {0, ..., 17}
    reward: float              # R(s_t, a_t)
    constraint_cost: float     # c(s_t, a_t) = 𝟙[ℓ_t > 50 ms]
    next_state: np.ndarray     # s_{t+1} ∈ ℝ¹¹
    done: bool                 # Always False for the continuing task
    log_prob: float            # log π_θ(a_t | s_t)
    value: float               # V_ψ(s_t)
    lyapunov_value: float      # L_φ(s_t)


# ── TelemetryRecord ───────────────────────────────────────────────────────────

@dataclass
class TelemetryRecord:
    """Flat record written to Parquet telemetry files once per frame."""

    frame_id: int
    timestamp: float

    # Stage latencies (ms)
    latency_ms: float
    latency_preprocess_ms: float
    latency_detect_ms: float
    latency_segment_ms: float       # 0.0 if stage 2 was skipped
    latency_postprocess_ms: float

    # Detection stats
    detection_count: int
    mean_confidence: float
    defect_area_ratio: float

    # Controller metadata
    controller_action: int
    resolution_active: int          # 320, 480, or 640
    segmentation_active: bool
    threshold_active: float

    # GPU metrics
    gpu_util_percent: float
    gpu_temp_celsius: float
    gpu_memory_used_mb: float

    # Conformal prediction
    conformal_upper_bound_ms: float
    conformal_alpha: float

    # Drift detection
    ks_p_value: float
    drift_alert: bool

    # RL / safety
    lyapunov_value: float
    constraint_cost: float          # 0.0 or 1.0
    reward: float
