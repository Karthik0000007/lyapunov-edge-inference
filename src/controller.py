"""
src/controller.py
─────────────────
Three-layer adaptive controller orchestrating:

    Layer 1: Lyapunov-PPO agent (RL action with safety masking)
    Layer 2: Conformal prediction override check
    Layer 3: Rule-based fallback (consecutive-violation tracking)

Plus config application logic with decision-frequency gating and
introspection logging.

Public API
──────────
    AdaptiveController
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from src.agent_lyapunov_ppo import LyapunovPPOAgent
from src.baselines import RuleBasedController
from src.conformal import ConformalPredictor
from src.state_features import ControllerAction, ControllerState

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_NUM_RESOLUTIONS: int = 3  # {0: 320, 1: 480, 2: 640}
_NUM_THRESHOLDS: int = 3  # {0: base, 1: +0.1, 2: +0.2}

# Most-conservative action: lowest resolution, highest threshold, seg off.
# Encoding: res_code=0, thr_code=2, seg=0 → 0*6 + 2*2 + 0 = 4.
_MAX_DEGRADATION_ACTION: int = 4


# ── Decision record ─────────────────────────────────────────────────────────


@dataclass
class ControllerDecision:
    """Introspection record for a single controller decision."""

    raw_action: int  # RL agent's proposed action
    safe_actions: List[int]  # Lyapunov safe set
    lyapunov_value: float  # L_φ(s)
    conformal_bound: float  # U_t (conformal upper bound)
    conformal_overridden: bool  # Whether conformal override fired
    fallback_active: bool  # Whether rule-based fallback is active
    final_action: int  # Action actually applied
    resolution_index: int  # Resulting resolution index
    threshold_index: int  # Resulting threshold index
    segmentation_enabled: bool  # Resulting segmentation flag


# ── AdaptiveController ───────────────────────────────────────────────────────


class AdaptiveController:
    """Three-layer adaptive controller.

    Sequentially applies:
    1. **Layer 1 — RL**: ``LyapunovPPOAgent.select_action()`` with Lyapunov
       masking produces a candidate action.
    2. **Layer 2 — Conformal**: ``ConformalPredictor.check_action()`` verifies
       the RL action will not breach the latency budget; overrides to a safe
       alternative if needed.
    3. **Layer 3 — Rule fallback**: ``RuleFallbackChecker`` tracks consecutive
       latency violations.  After ≥ ``violation_trigger`` consecutive violations,
       it forces the maximum-degradation config until ``recovery_window``
       consecutive violation-free frames are observed.

    Parameters
    ----------
    config:
        Full controller configuration dict (from ``config/controller.yaml``).
    device:
        Torch device (``cpu`` or ``cuda``).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device | None = None,
    ) -> None:
        self._device = device or torch.device("cpu")

        agent_cfg = config.get("agent", {})
        conformal_cfg = config.get("conformal", {})
        fallback_cfg = config.get("fallback", {})

        # Decision frequency gating: decide every k-th frame.
        self._decision_freq: int = agent_cfg.get("decision_frequency", 5)
        self._frame_counter: int = 0

        # ── Layer 1: RL agent ────────────────────────────────────────────
        self._agent = LyapunovPPOAgent(config, device=self._device)

        # ── Layer 2: Conformal prediction ────────────────────────────────
        self._conformal = ConformalPredictor(
            conformal_cfg,
            latency_budget_ms=50.0,
            device=self._device,
        )

        # ── Layer 3: Rule-based fallback ─────────────────────────────────
        self._fallback = _RuleFallbackChecker(
            violation_trigger=fallback_cfg.get("consecutive_violations", 3),
            recovery_window=fallback_cfg.get("recovery_window", 50),
        )

        # ── Pipeline config state ────────────────────────────────────────
        self._resolution_index: int = 2  # Start at max quality (640)
        self._threshold_index: int = 0  # Start at base threshold
        self._segmentation_enabled: bool = True

        # ── Held action (used between decision frames) ───────────────────
        self._held_action: Optional[ControllerAction] = None
        self._last_decision: Optional[ControllerDecision] = None

        logger.info(
            "AdaptiveController ready — decision_freq=%d  " "conformal=%s  fallback=%s",
            self._decision_freq,
            "ON" if self._conformal.enabled else "OFF",
            "ON" if fallback_cfg.get("enabled", True) else "OFF",
        )

    # ── Main entry point ─────────────────────────────────────────────────

    def step(
        self,
        state: ControllerState,
        observed_latency: float,
        latency_budget_ms: float = 50.0,
    ) -> ControllerAction:
        """Run the three-layer controller and return the action to apply.

        Parameters
        ----------
        state:
            Current 11-dimensional controller state.
        observed_latency:
            Latency observed on the *previous* frame (ms).  Used by the
            fallback layer to track violations.
        latency_budget_ms:
            Hard latency budget (default 50.0 ms).

        Returns
        -------
        ControllerAction
            Action to apply to the pipeline.
        """
        self._frame_counter += 1

        # Update fallback tracker with latest observed latency.
        violated = observed_latency > latency_budget_ms
        self._fallback.update(violated)

        # Decision-frequency gating: only decide every k-th frame.
        if self._frame_counter % self._decision_freq != 1 and self._held_action is not None:
            return self._held_action

        state_tensor = state.to_tensor()

        # ── Layer 1: RL agent ────────────────────────────────────────
        action_idx, log_prob, value, lyap_val = self._agent.select_action(state_tensor)
        safe_actions = self._agent.lyapunov.compute_safe_actions(state_tensor)

        # ── Layer 2: Conformal override ──────────────────────────────
        final_idx, conformal_bound, was_overridden = self._conformal.check_action(
            state_tensor, action_idx, safe_actions
        )

        # ── Layer 3: Rule-based fallback ─────────────────────────────
        fallback_active = self._fallback.is_active
        if fallback_active:
            final_idx = _MAX_DEGRADATION_ACTION
            logger.info(
                "Fallback ACTIVE — forcing max-degradation action %d "
                "(consecutive violations=%d, clean frames=%d)",
                _MAX_DEGRADATION_ACTION,
                self._fallback.consecutive_violations,
                self._fallback.clean_frames,
            )

        # ── Apply action to config ───────────────────────────────────
        action = ControllerAction.from_index(final_idx)
        self._apply_action(action)

        # ── Introspection logging ────────────────────────────────────
        self._last_decision = ControllerDecision(
            raw_action=action_idx,
            safe_actions=safe_actions,
            lyapunov_value=lyap_val,
            conformal_bound=conformal_bound,
            conformal_overridden=was_overridden,
            fallback_active=fallback_active,
            final_action=final_idx,
            resolution_index=self._resolution_index,
            threshold_index=self._threshold_index,
            segmentation_enabled=self._segmentation_enabled,
        )

        logger.debug(
            "Controller decision — raw=%d  safe_set=%s  L(s)=%.4f  "
            "U_t=%.2f ms  override=%s  fallback=%s  final=%d  "
            "res=%d  thr=%d  seg=%s",
            action_idx,
            safe_actions,
            lyap_val,
            conformal_bound,
            was_overridden,
            fallback_active,
            final_idx,
            self._resolution_index,
            self._threshold_index,
            self._segmentation_enabled,
        )

        # Hold action for non-decision frames.
        self._held_action = action
        return action

    # ── Config application ───────────────────────────────────────────────

    def _apply_action(self, action: ControllerAction) -> None:
        """Apply factored action deltas to pipeline config with clamping."""
        self._resolution_index = max(
            0, min(_NUM_RESOLUTIONS - 1, self._resolution_index + action.resolution_delta)
        )
        self._threshold_index = max(
            0, min(_NUM_THRESHOLDS - 1, self._threshold_index + action.threshold_delta)
        )
        self._segmentation_enabled = action.segmentation_enabled

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def resolution_index(self) -> int:
        return self._resolution_index

    @property
    def threshold_index(self) -> int:
        return self._threshold_index

    @property
    def segmentation_enabled(self) -> bool:
        return self._segmentation_enabled

    @property
    def last_decision(self) -> Optional[ControllerDecision]:
        return self._last_decision

    @property
    def agent(self) -> LyapunovPPOAgent:
        return self._agent

    @property
    def conformal(self) -> ConformalPredictor:
        return self._conformal

    @property
    def fallback(self) -> "_RuleFallbackChecker":
        return self._fallback


# ── Rule-based fallback checker ──────────────────────────────────────────────


class _RuleFallbackChecker:
    """Tracks consecutive latency violations and manages fallback state.

    Activation: triggers after ``violation_trigger`` consecutive frames that
    violate the latency budget.

    Recovery: once activated, remains active until ``recovery_window``
    consecutive violation-free frames are observed.

    Parameters
    ----------
    violation_trigger:
        Number of consecutive violations before activation (default 3).
    recovery_window:
        Number of violation-free frames required to exit fallback
        (default 50).
    """

    def __init__(self, violation_trigger: int = 3, recovery_window: int = 50) -> None:
        self._trigger: int = violation_trigger
        self._recovery: int = recovery_window

        self._consecutive_violations: int = 0
        self._clean_frames: int = 0
        self._active: bool = False

    def update(self, violated: bool) -> None:
        """Update the fallback tracker with a new frame outcome.

        Parameters
        ----------
        violated:
            ``True`` if the latest frame exceeded the latency budget.
        """
        if violated:
            self._consecutive_violations += 1
            self._clean_frames = 0

            # Activate if enough consecutive violations.
            if not self._active and self._consecutive_violations >= self._trigger:
                self._active = True
                logger.warning(
                    "Fallback ACTIVATED after %d consecutive violations",
                    self._consecutive_violations,
                )
        else:
            self._consecutive_violations = 0

            if self._active:
                self._clean_frames += 1
                # Deactivate after enough clean frames.
                if self._clean_frames >= self._recovery:
                    self._active = False
                    self._clean_frames = 0
                    logger.info(
                        "Fallback DEACTIVATED after %d clean frames",
                        self._recovery,
                    )

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def consecutive_violations(self) -> int:
        return self._consecutive_violations

    @property
    def clean_frames(self) -> int:
        return self._clean_frames

    def reset(self) -> None:
        """Reset fallback state."""
        self._consecutive_violations = 0
        self._clean_frames = 0
        self._active = False
