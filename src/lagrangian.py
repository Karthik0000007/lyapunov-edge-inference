"""
src/lagrangian.py
─────────────────
Lagrangian dual-variable mechanism for constrained policy optimisation.

Maintains the Lagrange multiplier λ and provides the augmented loss
computation for the PPO objective: L = J_R − λ · J_C.

Public API
──────────
    LagrangianDual
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


class LagrangianDual:
    """Lagrangian dual variable with dual gradient ascent.

    The augmented PPO loss is:

        L = J_R − λ · J_C

    λ is updated via dual gradient ascent:

        λ_{k+1} = max(0, λ_k + η_λ · (J_C − d_0))

    Parameters
    ----------
    config:
        ``lagrangian`` section from ``config/controller.yaml``.
        Expected keys::

            lambda_init: 0.1
            lambda_lr: 0.01
            constraint_threshold: 0.01
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._lambda: float = config.get("lambda_init", 0.1)
        self._lr: float = config.get("lambda_lr", 0.01)
        self._threshold: float = config.get("constraint_threshold", 0.01)

        logger.info(
            "LagrangianDual initialised — λ₀=%.4f  η_λ=%.4f  d₀=%.4f",
            self._lambda,
            self._lr,
            self._threshold,
        )

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def lambda_value(self) -> float:
        """Current Lagrange multiplier λ."""
        return self._lambda

    @property
    def threshold(self) -> float:
        """Constraint budget d₀ (max acceptable violation rate)."""
        return self._threshold

    # ── Dual update ──────────────────────────────────────────────────────

    def update(self, constraint_cost: float) -> float:
        """Dual gradient ascent step.

        λ_{k+1} = max(0, λ_k + η_λ · (J_C − d_0))

        Parameters
        ----------
        constraint_cost:
            Empirical constraint cost J_C (mean violation rate over the
            current batch).

        Returns
        -------
        float
            Updated λ value.
        """
        self._lambda = max(
            0.0, self._lambda + self._lr * (constraint_cost - self._threshold)
        )
        return self._lambda

    # ── Augmented loss ───────────────────────────────────────────────────

    def augmented_loss(
        self,
        policy_loss: torch.Tensor,
        constraint_cost: torch.Tensor,
    ) -> torch.Tensor:
        """Compute augmented PPO loss: L = J_R − λ · J_C.

        Parameters
        ----------
        policy_loss:
            PPO clipped surrogate loss (to be minimised, so positive means
            worse).  This corresponds to −J_R.
        constraint_cost:
            Mean constraint cost over the batch (differentiable).

        Returns
        -------
        torch.Tensor
            Augmented scalar loss.
        """
        return policy_loss + self._lambda * constraint_cost

    # ── State I/O ────────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, float]:
        """Serialise dual state."""
        return {
            "lambda": self._lambda,
            "lr": self._lr,
            "threshold": self._threshold,
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        """Restore dual state."""
        self._lambda = state["lambda"]
        self._lr = state.get("lr", self._lr)
        self._threshold = state.get("threshold", self._threshold)
        logger.info("Restored LagrangianDual — λ=%.6f", self._lambda)
