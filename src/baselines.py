"""
src/baselines.py
────────────────
Baseline controllers for evaluation and ablation:

    BaseController   — abstract interface
    FixedHighQualityController — 640, seg=on, base threshold
    FixedLowLatencyController  — 320, seg=off, max threshold
    RuleBasedController        — heuristic P99-based switching
    PIDController              — proportional control on latency error
    CONTROLLER_REGISTRY        — name → class mapping

Public API
──────────
    BaseController, FixedHighQualityController, FixedLowLatencyController,
    RuleBasedController, PIDController, CONTROLLER_REGISTRY
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional, Type

from src.state_features import ControllerAction, ControllerState, Transition

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_NUM_ACTIONS: int = 18
_NUM_RESOLUTIONS: int = 3  # {0: 320, 1: 480, 2: 640}
_NUM_THRESHOLDS: int = 3  # {0: base, 1: +0.1, 2: +0.2}


# ── Abstract base ───────────────────────────────────────────────────────────


class BaseController(abc.ABC):
    """Abstract controller interface.

    Every controller must implement:
    - ``select_action(state)`` → ``ControllerAction``
    - ``update(transition)`` → ``None``  (for online learning; no-op for baselines)
    """

    @abc.abstractmethod
    def select_action(self, state: ControllerState) -> ControllerAction:
        """Choose a pipeline configuration action given the current state.

        Parameters
        ----------
        state:
            Current 11-dimensional controller state.

        Returns
        -------
        ControllerAction
            Action encoding resolution delta, threshold delta, and
            segmentation flag.
        """

    def update(self, transition: Transition) -> None:
        """Ingest a transition for online learning.  No-op by default."""


# ── Fixed controllers ────────────────────────────────────────────────────────


class FixedHighQualityController(BaseController):
    """Always selects maximum quality: 640, seg=on, base threshold.

    Action encoding: res_delta=0, thr_delta=0, seg=1 (hold at highest).
    This works correctly only when pipeline is already at max — the
    controller always requests "keep current + seg on".  The action
    encoding for hold-high is: res_code=1, thr_code=1, seg=1 → index 9.
    But since deltas are clamped, we use res=+1, thr=-1, seg=1 to
    push towards max resolution, min threshold, seg on.
    """

    def select_action(self, state: ControllerState) -> ControllerAction:
        # Push towards 640 (res_delta=+1), base threshold (thr_delta=-1), seg=on.
        # Encoding: res_code=2, thr_code=0, seg=1 → 2*6 + 0*2 + 1 = 13.
        return ControllerAction.from_index(13)


class FixedLowLatencyController(BaseController):
    """Always selects minimum latency: 320, seg=off, max threshold.

    Pushes towards lowest resolution (res_delta=-1), highest threshold
    (thr_delta=+1), segmentation off.
    """

    def select_action(self, state: ControllerState) -> ControllerAction:
        # Push towards 320 (res_delta=-1), max threshold (thr_delta=+1), seg=off.
        # Encoding: res_code=0, thr_code=2, seg=0 → 0*6 + 2*2 + 0 = 4.
        return ControllerAction.from_index(4)


# ── Rule-based controller ───────────────────────────────────────────────────


class RuleBasedController(BaseController):
    """Heuristic controller based on windowed P99 latency thresholds.

    Rules
    -----
    - If P99 > 45 ms: decrease resolution and/or disable segmentation.
    - If P99 > 40 ms: hold or slightly degrade (increase threshold).
    - If P99 < 35 ms: increase quality (resolution, enable segmentation).
    - Otherwise: hold current configuration.
    """

    def __init__(
        self,
        high_threshold_ms: float = 45.0,
        mid_threshold_ms: float = 40.0,
        low_threshold_ms: float = 35.0,
    ) -> None:
        self._high = high_threshold_ms
        self._mid = mid_threshold_ms
        self._low = low_threshold_ms

    def select_action(self, state: ControllerState) -> ControllerAction:
        p99 = state.p99_latency_ms

        if p99 > self._high:
            # Aggressive degradation: lower resolution, raise threshold, seg off.
            return ControllerAction.from_index(self._encode(-1, 1, False))

        if p99 > self._mid:
            # Mild degradation: raise threshold only, keep resolution.
            return ControllerAction.from_index(self._encode(0, 1, False))

        if p99 < self._low:
            # Upgrade: raise resolution, lower threshold, enable seg.
            return ControllerAction.from_index(self._encode(1, -1, True))

        # Hold current configuration.
        seg_on = bool(state.segmentation_enabled)
        return ControllerAction.from_index(self._encode(0, 0, seg_on))

    @staticmethod
    def _encode(res_delta: int, thr_delta: int, seg: bool) -> int:
        """Encode factored action into flat index."""
        res_code = res_delta + 1  # {-1, 0, +1} → {0, 1, 2}
        thr_code = thr_delta + 1
        seg_code = int(seg)
        return res_code * 6 + thr_code * 2 + seg_code


# ── PID controller ──────────────────────────────────────────────────────────


class PIDController(BaseController):
    """Proportional controller mapping latency error to action deltas.

    The error signal is  e_t = last_latency - target.

    - If e_t > 0 (over budget): degrade proportionally.
    - If e_t < 0 (under budget): upgrade proportionally.

    Only the proportional term is used (no integral / derivative) because
    the system dynamics change faster than a PID can track with stable
    gains.

    Parameters
    ----------
    target_ms:
        Latency target (default 50.0 ms, matching the pipeline budget).
    kp:
        Proportional gain.  Higher → more aggressive response.
    """

    def __init__(self, target_ms: float = 50.0, kp: float = 0.1) -> None:
        self._target = target_ms
        self._kp = kp

    def select_action(self, state: ControllerState) -> ControllerAction:
        error = state.last_latency_ms - self._target
        # Proportional signal → map to discrete deltas.
        signal = self._kp * error

        # Resolution delta.
        if signal > 0.5:
            res_delta = -1  # Over budget → decrease resolution
        elif signal < -0.5:
            res_delta = 1  # Under budget → increase resolution
        else:
            res_delta = 0

        # Threshold delta (proportional, opposite direction).
        if signal > 0.3:
            thr_delta = 1  # Over budget → raise threshold (fewer detections)
        elif signal < -0.3:
            thr_delta = -1  # Under budget → lower threshold
        else:
            thr_delta = 0

        # Segmentation: disable when over budget, enable when well under.
        seg = signal < -0.2

        res_code = res_delta + 1
        thr_code = thr_delta + 1
        seg_code = int(seg)
        action_index = res_code * 6 + thr_code * 2 + seg_code

        return ControllerAction.from_index(action_index)


# ── Controller registry ─────────────────────────────────────────────────────

CONTROLLER_REGISTRY: Dict[str, Type[BaseController]] = {
    "fixed_high_quality": FixedHighQualityController,
    "fixed_low_latency": FixedLowLatencyController,
    "rule_based": RuleBasedController,
    "pid": PIDController,
}
