"""
src/conformal.py
────────────────
Conformal prediction safety certificate with Adaptive Conformal Inference
(ACI) for online α-adaptation under distribution shift.

Provides distribution-free P99 latency upper bounds that complement the
Lyapunov safety layer with finite-sample coverage guarantees.

Public API
──────────
    ConformalPredictor
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.latency_predictor import LatencyPredictor
from src.torch_compat import torch_load_compat

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_NUM_ACTIONS: int = 18
_MOST_CONSERVATIVE_ACTION: int = 4
# Action 4: res_delta=-1, thr_delta=+1, seg=0  (lowest resolution, highest
# threshold, segmentation off).  Encoding: res_code=0, thr_code=2, seg=0
# → 0*6 + 2*2 + 0 = 4.


# ── ConformalPredictor ───────────────────────────────────────────────────────


class ConformalPredictor:
    """Conformal prediction system with ACI online adaptation.

    Parameters
    ----------
    config:
        ``conformal`` section from ``config/controller.yaml``.  Expected keys::

            enabled: true
            alpha_target: 0.01
            alpha_lr: 0.005
            calibration_size: 5000
            predictor_hidden_size: 32
            predictor_checkpoint: checkpoints/conformal/latency_predictor.pt
    latency_budget_ms:
        Hard latency budget (default 50.0 ms).
    device:
        Torch device.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        latency_budget_ms: float = 50.0,
        device: torch.device | None = None,
    ) -> None:
        self._device = device or torch.device("cpu")
        self.enabled: bool = config.get("enabled", True)
        self._budget_ms: float = latency_budget_ms

        # ACI hyper-parameters.
        self._alpha_target: float = config.get("alpha_target", 0.01)
        self._alpha: float = self._alpha_target  # Current adaptive α
        self._alpha_lr: float = config.get("alpha_lr", 0.005)
        self._calibration_size: int = config.get("calibration_size", 5000)

        # Calibration quantile (set during calibrate()).
        self._quantile: float = 0.0
        self._calibrated: bool = False

        # Rolling calibration set for online re-calibration.
        self._scores: deque[float] = deque(maxlen=self._calibration_size)

        # Latency predictor (shared config section).
        self._predictor = LatencyPredictor(config, device=self._device)

        logger.info(
            "ConformalPredictor initialised — α_target=%.4f  lr=%.4f  "
            "calibration_size=%d  budget=%.1f ms",
            self._alpha_target,
            self._alpha_lr,
            self._calibration_size,
            self._budget_ms,
        )

    # ── Offline calibration ──────────────────────────────────────────────

    def calibrate(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        latencies: np.ndarray,
    ) -> float:
        """Offline calibration: compute nonconformity scores and quantile.

        Parameters
        ----------
        states:
            ``(N, 11)`` normalised states.
        actions:
            ``(N,)`` integer actions.
        latencies:
            ``(N,)`` observed latencies (ms).

        Returns
        -------
        float
            Calibration quantile q̂_{1-α}.
        """
        n = len(states)
        s_t = torch.tensor(states, dtype=torch.float32, device=self._device)
        a_t = torch.tensor(actions, dtype=torch.long, device=self._device)

        # Predict latencies for all calibration samples.
        predicted = self._predictor.predict_batch(s_t, a_t).cpu().numpy()

        # Nonconformity scores: residuals e_i = ℓ_i − ℓ̂(s_i, a_i).
        scores = latencies - predicted

        # Store in rolling calibration set.
        self._scores.clear()
        self._scores.extend(scores.tolist())

        # Extract quantile at (1 − α).
        self._quantile = float(np.quantile(scores, 1.0 - self._alpha_target))
        self._calibrated = True

        logger.info(
            "Calibrated on %d samples — quantile q̂_{1-α}=%.4f ms  " "(α=%.4f)",
            n,
            self._quantile,
            self._alpha_target,
        )
        return self._quantile

    # ── Prediction bound ─────────────────────────────────────────────────

    def predict_bound(self, state: torch.Tensor, action: int) -> float:
        """Compute the conformal upper bound U_t = ℓ̂(s, a) + q̂.

        Parameters
        ----------
        state:
            Normalised state tensor ``(11,)``.
        action:
            Action index ``[0, 17]``.

        Returns
        -------
        float
            Upper bound on latency (ms).
        """
        predicted = self._predictor.predict(state, action)
        return predicted + self._quantile

    # ── Override decision ────────────────────────────────────────────────

    def check_action(
        self,
        state: torch.Tensor,
        proposed_action: int,
        safe_actions: Optional[List[int]] = None,
    ) -> Tuple[int, float, bool]:
        """Evaluate whether the proposed action satisfies the latency bound.

        Parameters
        ----------
        state:
            Normalised state ``(11,)``.
        proposed_action:
            Action selected by the RL agent.
        safe_actions:
            Lyapunov-safe action set (if available).

        Returns
        -------
        Tuple[int, float, bool]
            ``(final_action, upper_bound_ms, was_overridden)``.
        """
        if not self.enabled or not self._calibrated:
            bound = self.predict_bound(state, proposed_action)
            return proposed_action, bound, False

        bound = self.predict_bound(state, proposed_action)

        if bound <= self._budget_ms:
            return proposed_action, bound, False

        # Override: try alternatives from safe set first.
        logger.info(
            "Conformal override: U_t=%.2f ms > budget=%.1f ms for action %d",
            bound,
            self._budget_ms,
            proposed_action,
        )

        candidates = safe_actions if safe_actions else list(range(_NUM_ACTIONS))
        best_action = proposed_action
        best_bound = bound

        for a in candidates:
            if a == proposed_action:
                continue
            a_bound = self.predict_bound(state, a)
            if a_bound <= self._budget_ms and a_bound < best_bound:
                best_action = a
                best_bound = a_bound

        # If no safe candidate found, use most conservative action.
        if best_bound > self._budget_ms:
            best_action = _MOST_CONSERVATIVE_ACTION
            best_bound = self.predict_bound(state, best_action)

        return best_action, best_bound, True

    # ── ACI online update ────────────────────────────────────────────────

    def update(self, state: torch.Tensor, action: int, observed_latency: float) -> None:
        """Online ACI update after observing the true latency.

        Updates:
        1. The rolling calibration set with the new nonconformity score.
        2. α via the ACI rule: α_{t+1} = α_t + η(α_target − err_t).
        3. The quantile from the updated calibration set.

        Parameters
        ----------
        state:
            Normalised state ``(11,)``.
        action:
            Action that was executed.
        observed_latency:
            Actual latency observed (ms).
        """
        predicted = self._predictor.predict(state, action)
        score = observed_latency - predicted

        # Was the bound violated?
        err_t = 1.0 if observed_latency > (predicted + self._quantile) else 0.0

        # ACI update: α_{t+1} = α_t + η·(α_target − err_t)
        self._alpha = self._alpha + self._alpha_lr * (self._alpha_target - err_t)
        self._alpha = max(1e-6, min(1.0 - 1e-6, self._alpha))

        # Add score to rolling calibration set.
        self._scores.append(score)

        # Recompute quantile from updated calibration set.
        if len(self._scores) > 0:
            self._quantile = float(np.quantile(list(self._scores), 1.0 - self._alpha))

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        """Current adaptive α."""
        return self._alpha

    @property
    def quantile(self) -> float:
        """Current calibration quantile q̂."""
        return self._quantile

    @property
    def predictor(self) -> LatencyPredictor:
        """Access the underlying latency predictor."""
        return self._predictor

    # ── Checkpoint I/O ───────────────────────────────────────────────────

    def save_state(self, path: Path) -> None:
        """Serialise conformal state (α, quantile, scores) to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "alpha": self._alpha,
            "quantile": self._quantile,
            "calibrated": self._calibrated,
            "scores": list(self._scores),
        }
        torch.save(state, path)
        logger.info("Saved conformal state to %s", path)

    def load_state(self, path: Path) -> None:
        """Restore conformal state from disk."""
        data = torch_load_compat(path, map_location="cpu")
        self._alpha = data["alpha"]
        self._quantile = data["quantile"]
        self._calibrated = data["calibrated"]
        self._scores.clear()
        self._scores.extend(data["scores"])
        logger.info(
            "Loaded conformal state from %s — α=%.6f  q̂=%.4f  scores=%d",
            path,
            self._alpha,
            self._quantile,
            len(self._scores),
        )
