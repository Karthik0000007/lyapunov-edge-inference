"""
src/env.py
──────────
Gymnasium environment for offline RL training on logged telemetry traces.

The environment replays Parquet trace files and simulates the effect of
adaptive controller actions on the pipeline configuration.  Latency is
interpolated from recorded data based on the active config.

Public API
──────────
    LatencyEnv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_STATE_DIM: int = 11
_NUM_ACTIONS: int = 18
_NUM_RESOLUTIONS: int = 3
_NUM_THRESHOLDS: int = 3

_LATENCY_BUDGET_MS: float = 50.0

# Resolution map: index → pixel width.
_RESOLUTION_MAP: Dict[int, int] = {0: 320, 1: 480, 2: 640}

# Normalization ranges (matching state_features.py).
_LATENCY_MIN: float = 5.0
_LATENCY_MAX: float = 100.0
_DET_COUNT_MAX: float = 50.0
_GPU_UTIL_MAX: float = 100.0
_GPU_TEMP_MIN: float = 30.0
_GPU_TEMP_MAX: float = 100.0


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _normalize(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return _clamp01((v - lo) / (hi - lo))


# ── LatencyEnv ───────────────────────────────────────────────────────────────

class LatencyEnv(gym.Env):
    """Gymnasium environment replaying logged Parquet traces.

    Observation space: ``Box(0, 1, shape=(11,), dtype=float32)``
    Action space: ``Discrete(18)``

    The 11-dimensional observation follows the same layout as
    ``ControllerState``:

    ===========  ===============================================
    Index        Feature
    ===========  ===============================================
    0            last latency (normalised)
    1            windowed mean latency (normalised)
    2            windowed P99 latency (normalised)
    3            detection count (normalised)
    4            mean confidence
    5            defect area ratio
    6            resolution index (0.0 / 0.5 / 1.0)
    7            threshold index (0.0 / 0.5 / 1.0)
    8            segmentation enabled (0 or 1)
    9            GPU utilization (normalised)
    10           GPU temperature (normalised)
    ===========  ===============================================

    Parameters
    ----------
    trace_path:
        Path to a Parquet file with telemetry columns (see below).
    max_steps:
        Episode truncation length (default 1000).
    latency_budget_ms:
        Hard latency budget for constraint cost computation (default 50.0).
    quality_weight:
        Reward weight for detection quality (default 1.0).
    churn_penalty:
        Reward penalty coefficient for action churn (default 0.05).
    seed:
        Optional random seed.

    Required Parquet columns
    ────────────────────────
    ``latency_ms``, ``detection_count``, ``mean_confidence``,
    ``defect_area_ratio``, ``gpu_util_percent``, ``gpu_temp_celsius``,
    ``resolution_active``, ``segmentation_active``, ``threshold_active``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trace_path: str | Path = "data/telemetry.parquet",
        max_steps: int = 1000,
        latency_budget_ms: float = 50.0,
        quality_weight: float = 1.0,
        churn_penalty: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # ── Spaces ───────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(_STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(_NUM_ACTIONS)

        # ── Config ───────────────────────────────────────────────────
        self._max_steps: int = max_steps
        self._budget: float = latency_budget_ms
        self._quality_weight: float = quality_weight
        self._churn_penalty: float = churn_penalty

        # ── Load trace ───────────────────────────────────────────────
        self._trace_path = Path(trace_path)
        self._trace: pd.DataFrame = self._load_trace(self._trace_path)
        self._trace_len: int = len(self._trace)

        # ── Episode state ────────────────────────────────────────────
        self._step_count: int = 0
        self._offset: int = 0    # Random start offset into trace

        self._resolution_index: int = 2
        self._threshold_index: int = 0
        self._segmentation_enabled: int = 1

        self._prev_action: int = -1
        self._latency_window: list[float] = []
        self._window_size: int = 50

        self._np_random: np.random.Generator = np.random.default_rng(seed)

        logger.info(
            "LatencyEnv ready — trace=%s  rows=%d  max_steps=%d  "
            "budget=%.1f ms",
            self._trace_path,
            self._trace_len,
            self._max_steps,
            self._budget,
        )

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment with a random trace offset.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            ``(observation, info)``
        """
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Random offset for episode diversity.
        max_offset = max(0, self._trace_len - self._max_steps - 1)
        self._offset = int(self._np_random.integers(0, max_offset + 1))

        # Reset pipeline config.
        self._resolution_index = 2
        self._threshold_index = 0
        self._segmentation_enabled = 1

        self._step_count = 0
        self._prev_action = -1
        self._latency_window.clear()

        obs = self._build_observation()
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step: apply action, look up latency, compute reward.

        Parameters
        ----------
        action:
            Discrete action index in ``[0, 17]``.

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
            ``(observation, reward, terminated, truncated, info)``
        """
        assert self.action_space.contains(action), (
            f"Invalid action {action}; must be in [0, {_NUM_ACTIONS - 1}]"
        )

        # Decode factored action.
        seg = action % 2
        thr_delta = (action // 2) % 3 - 1
        res_delta = (action // 6) % 3 - 1

        # Apply deltas with clamping.
        self._resolution_index = max(
            0, min(_NUM_RESOLUTIONS - 1, self._resolution_index + res_delta)
        )
        self._threshold_index = max(
            0, min(_NUM_THRESHOLDS - 1, self._threshold_index + thr_delta)
        )
        self._segmentation_enabled = seg

        # Look up latency from trace for current config.
        trace_idx = (self._offset + self._step_count) % self._trace_len
        base_latency = self._lookup_latency(trace_idx)
        latency = self._interpolate_latency(base_latency)

        # Update latency window.
        self._latency_window.append(latency)
        if len(self._latency_window) > self._window_size:
            self._latency_window = self._latency_window[-self._window_size:]

        # ── Reward: Q_t - 0.05 * ||a_t - a_{t-1}||_1 ────────────────
        quality = self._compute_quality(trace_idx)
        if self._prev_action >= 0:
            churn = self._action_l1_distance(action, self._prev_action)
        else:
            churn = 0.0
        reward = self._quality_weight * quality - self._churn_penalty * churn

        # ── Constraint cost: 1[ℓ > budget] ──────────────────────────
        constraint_cost = 1.0 if latency > self._budget else 0.0

        self._prev_action = action
        self._step_count += 1

        terminated = False
        truncated = self._step_count >= self._max_steps

        obs = self._build_observation()

        info: Dict[str, Any] = {
            "constraint_cost": constraint_cost,
            "latency_ms": latency,
            "quality": quality,
            "churn": churn,
            "resolution_index": self._resolution_index,
            "threshold_index": self._threshold_index,
            "segmentation_enabled": bool(self._segmentation_enabled),
        }

        return obs, reward, terminated, truncated, info

    # ── Observation assembly ─────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """Build normalised 11-d observation from current state."""
        trace_idx = (self._offset + self._step_count) % self._trace_len
        row = self._trace.iloc[trace_idx]

        # Latency features.
        if len(self._latency_window) > 0:
            last_lat = self._latency_window[-1]
            mean_lat = float(np.mean(self._latency_window))
            p99_lat = float(np.percentile(self._latency_window, 99))
        else:
            last_lat = float(row.get("latency_ms", 30.0))
            mean_lat = last_lat
            p99_lat = last_lat

        det_count = float(row.get("detection_count", 0))
        mean_conf = float(row.get("mean_confidence", 0.0))
        area_ratio = float(row.get("defect_area_ratio", 0.0))

        gpu_util = float(row.get("gpu_util_percent", 50.0))
        gpu_temp = float(row.get("gpu_temp_celsius", 55.0))

        obs = np.array([
            _normalize(last_lat, _LATENCY_MIN, _LATENCY_MAX),
            _normalize(mean_lat, _LATENCY_MIN, _LATENCY_MAX),
            _normalize(p99_lat, _LATENCY_MIN, _LATENCY_MAX),
            _normalize(det_count, 0.0, _DET_COUNT_MAX),
            _clamp01(mean_conf),
            _clamp01(area_ratio),
            _clamp01(self._resolution_index / 2.0),
            _clamp01(self._threshold_index / 2.0),
            float(self._segmentation_enabled),
            _normalize(gpu_util, 0.0, _GPU_UTIL_MAX),
            _normalize(gpu_temp, _GPU_TEMP_MIN, _GPU_TEMP_MAX),
        ], dtype=np.float32)

        return obs

    # ── Latency lookup and interpolation ─────────────────────────────────

    def _lookup_latency(self, trace_idx: int) -> float:
        """Get base latency from the trace at given index."""
        return float(self._trace.iloc[trace_idx].get("latency_ms", 30.0))

    def _interpolate_latency(self, base_latency: float) -> float:
        """Adjust base latency based on current config vs. recorded config.

        Higher resolution and segmentation-on increase latency; lower
        resolution and segmentation-off decrease it.  The scaling is a
        simple linear interpolation for offline replay.
        """
        # Resolution scaling: 320→0.7x, 480→1.0x, 640→1.3x
        res_scale = 0.7 + 0.3 * self._resolution_index

        # Segmentation overhead: ~15% when enabled.
        seg_scale = 1.15 if self._segmentation_enabled else 1.0

        # Threshold doesn't significantly affect latency.
        return base_latency * res_scale * seg_scale

    # ── Quality computation ──────────────────────────────────────────────

    def _compute_quality(self, trace_idx: int) -> float:
        """Proxy quality metric Q_t ∈ [0, 1] based on config and detections.

        Quality increases with resolution and lower threshold (more
        detections), and with segmentation enabled.
        """
        row = self._trace.iloc[trace_idx]
        det_count = float(row.get("detection_count", 0))
        mean_conf = float(row.get("mean_confidence", 0.0))

        # Resolution quality component: higher res → better quality.
        res_quality = self._resolution_index / 2.0

        # Threshold quality: lower threshold → better precision.
        thr_quality = 1.0 - self._threshold_index / 2.0

        # Segmentation bonus.
        seg_quality = 0.1 if self._segmentation_enabled else 0.0

        # Detection confidence contribution.
        conf_quality = mean_conf * 0.3

        quality = 0.3 * res_quality + 0.2 * thr_quality + seg_quality + conf_quality
        return _clamp01(quality)

    # ── Action distance ──────────────────────────────────────────────────

    @staticmethod
    def _action_l1_distance(a1: int, a2: int) -> float:
        """Compute L1 distance between two factored actions.

        Each action encodes (res_delta, thr_delta, seg) so the L1 distance
        is the sum of absolute differences of the factored components.
        """
        seg1, seg2 = a1 % 2, a2 % 2
        thr1, thr2 = (a1 // 2) % 3 - 1, (a2 // 2) % 3 - 1
        res1, res2 = (a1 // 6) % 3 - 1, (a2 // 6) % 3 - 1

        return float(
            abs(res1 - res2) + abs(thr1 - thr2) + abs(seg1 - seg2)
        )

    # ── Trace loading ────────────────────────────────────────────────────

    @staticmethod
    def _load_trace(path: Path) -> pd.DataFrame:
        """Load a Parquet trace file.

        Falls back to a synthetic trace when the file does not exist, so
        the environment can be tested without real data.
        """
        if path.exists():
            df = pd.read_parquet(path)
            logger.info("Loaded trace from %s — %d rows", path, len(df))
            return df

        logger.warning(
            "Trace file %s not found; generating synthetic trace "
            "(1000 rows) for development",
            path,
        )
        return _generate_synthetic_trace(n=1000)


# ── Synthetic trace generator ────────────────────────────────────────────────

def _generate_synthetic_trace(
    n: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Generate a synthetic telemetry trace for testing."""
    rng = np.random.default_rng(seed)

    latencies = 25.0 + 15.0 * rng.standard_normal(n)
    latencies = np.clip(latencies, 5.0, 100.0)

    det_counts = rng.poisson(5, size=n)
    confidences = rng.beta(5, 2, size=n)
    area_ratios = rng.beta(2, 20, size=n)

    gpu_utils = 40.0 + 30.0 * rng.random(n)
    gpu_temps = 45.0 + 20.0 * rng.random(n)

    resolutions = rng.choice([320, 480, 640], size=n)
    seg_flags = rng.choice([True, False], size=n)
    thresholds = rng.choice([0.3, 0.4, 0.5], size=n)

    return pd.DataFrame({
        "latency_ms": latencies,
        "detection_count": det_counts,
        "mean_confidence": confidences,
        "defect_area_ratio": area_ratios,
        "gpu_util_percent": gpu_utils,
        "gpu_temp_celsius": gpu_temps,
        "resolution_active": resolutions,
        "segmentation_active": seg_flags,
        "threshold_active": thresholds,
    })
