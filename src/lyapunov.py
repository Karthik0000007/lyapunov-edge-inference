"""
src/lyapunov.py
───────────────
Lyapunov safety layer: critic network, learned transition model, and
drift-based safe-action computation for per-step constraint enforcement.

Architecture
────────────
    Lyapunov Critic L_φ(s):
        Linear(11, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 1)
        Parameters: 4,865

    Transition Model T(s, a):
        Linear(29, 64) → ReLU → Linear(64, 11)
        Parameters: ~2,555

Public API
──────────
    LyapunovCritic, TransitionModel, compute_safe_actions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_STATE_DIM: int = 11
_NUM_ACTIONS: int = 18
_INPUT_DIM_TRANSITION: int = _STATE_DIM + _NUM_ACTIONS  # 29


# ── Lyapunov Critic ─────────────────────────────────────────────────────────

class LyapunovCritic(nn.Module):
    """Lyapunov critic L_φ(s): predicts discounted cumulative constraint cost.

    Architecture: Linear(11, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 1)
    """

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_STATE_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        state:
            ``(B, 11)`` or ``(11,)`` normalised state tensor.

        Returns
        -------
        torch.Tensor
            ``(B, 1)`` or ``(1,)`` Lyapunov value.
        """
        return self.net(state)


# ── Transition Model ─────────────────────────────────────────────────────────

class TransitionModel(nn.Module):
    """Tiny MLP predicting next state from (state, action).

    Input : concat(s ∈ ℝ¹¹, one_hot(a) ∈ ℝ¹⁸) → ℝ²⁹
    Hidden: Linear(29, 64) → ReLU
    Output: Linear(64, 11) → ŝ'
    """

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_INPUT_DIM_TRANSITION, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, _STATE_DIM),
        )

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Predict next state.

        Parameters
        ----------
        state:
            ``(B, 11)`` normalised state batch.
        action_onehot:
            ``(B, 18)`` one-hot encoded action batch.

        Returns
        -------
        torch.Tensor
            ``(B, 11)`` predicted next state.
        """
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x)

    def predict_single(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """Predict next state for a single (state, action) pair.

        Parameters
        ----------
        state:
            ``(11,)`` state vector.
        action:
            Integer action index.

        Returns
        -------
        torch.Tensor
            ``(11,)`` predicted next state.
        """
        s = state.unsqueeze(0) if state.dim() == 1 else state
        oh = torch.zeros(1, _NUM_ACTIONS, device=state.device)
        oh[0, action] = 1.0
        return self.forward(s, oh).squeeze(0)


# ── Lyapunov Manager ─────────────────────────────────────────────────────────

class LyapunovManager:
    """Manages the Lyapunov critic, transition model, training, and
    safe-action computation.

    Parameters
    ----------
    config:
        ``lyapunov`` section from ``config/controller.yaml``.  Expected keys::

            enabled: true
            critic_lr: 3.0e-4
            drift_tolerance: 0.0
    ppo_config:
        ``ppo`` section for ``gamma`` and ``hidden_size``.
    device:
        Torch device.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ppo_config: Dict[str, Any],
        device: torch.device | None = None,
    ) -> None:
        self._device = device or torch.device("cpu")
        self.enabled: bool = config.get("enabled", True)
        self._gamma: float = ppo_config.get("gamma", 0.99)
        self._drift_tol: float = config.get("drift_tolerance", 0.0)
        hidden = ppo_config.get("hidden_size", 64)
        lr = config.get("critic_lr", 3e-4)

        # Networks.
        self.critic = LyapunovCritic(hidden_size=hidden).to(self._device)
        self.transition = TransitionModel(hidden_size=hidden).to(self._device)

        # Optimizers.
        self._critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self._trans_optim = optim.Adam(self.transition.parameters(), lr=lr)

        # NaN detection flag.
        self._nan_detected: bool = False

        logger.info(
            "LyapunovManager initialised — %s  critic=%d params  "
            "transition=%d params  γ=%.2f  drift_tol=%.3f",
            "ENABLED" if self.enabled else "DISABLED",
            sum(p.numel() for p in self.critic.parameters()),
            sum(p.numel() for p in self.transition.parameters()),
            self._gamma,
            self._drift_tol,
        )

    # ── Safe-action computation ──────────────────────────────────────────

    @torch.no_grad()
    def compute_safe_actions(
        self,
        state: torch.Tensor,
        cost_estimates: Optional[List[float]] = None,
    ) -> List[int]:
        """Evaluate drift condition for all 18 actions and return safe set.

        Drift condition for action a:
            c(s, a) + γ · L_φ(ŝ') ≤ L_φ(s) + drift_tolerance

        Parameters
        ----------
        state:
            Normalised state ``(11,)`` tensor.
        cost_estimates:
            Optional per-action instantaneous cost estimates. If ``None``,
            costs are estimated from state features (latency above budget
            proxy).

        Returns
        -------
        List[int]
            A_safe ⊆ {0, ..., 17}.  Guaranteed non-empty (fallback to
            minimum-drift action).
        """
        if not self.enabled or self._nan_detected:
            return list(range(_NUM_ACTIONS))

        self.critic.eval()
        self.transition.eval()

        s = state.to(self._device)
        if s.dim() == 1:
            s = s.unsqueeze(0)

        L_s = self.critic(s).item()

        # Check for NaN in Lyapunov value.
        if not np.isfinite(L_s):
            logger.warning(
                "NaN/Inf in Lyapunov critic output; falling back to "
                "unrestricted action set"
            )
            self._nan_detected = True
            return list(range(_NUM_ACTIONS))

        safe_actions: List[int] = []
        drifts: List[float] = []

        for a in range(_NUM_ACTIONS):
            # Instantaneous cost estimate.
            if cost_estimates is not None:
                c_sa = cost_estimates[a]
            else:
                c_sa = self._estimate_cost(state, a)

            # Predict next state.
            s_next = self.transition.predict_single(s.squeeze(0), a)
            L_s_next = self.critic(s_next.unsqueeze(0)).item()

            if not np.isfinite(L_s_next):
                drifts.append(float("inf"))
                continue

            drift = c_sa + self._gamma * L_s_next - L_s
            drifts.append(drift)

            if drift <= self._drift_tol:
                safe_actions.append(a)

        # Fallback: pick action with minimum drift.
        if not safe_actions:
            min_drift_action = int(np.argmin(drifts))
            safe_actions = [min_drift_action]
            logger.debug(
                "Empty safe set — fallback to min-drift action %d (drift=%.4f)",
                min_drift_action,
                drifts[min_drift_action],
            )

        return safe_actions

    @staticmethod
    def _estimate_cost(state: torch.Tensor, action: int) -> float:
        """Heuristic cost estimate: higher cost for higher-resolution
        actions when latency features suggest overload."""
        s = state.squeeze()
        # Normalised P99 latency is at index 2.
        p99_norm = float(s[2].item()) if s.dim() > 0 else 0.0
        # Higher-resolution actions → more likely to violate budget.
        from src.state_features import ControllerAction

        ca = ControllerAction.from_index(action)
        # If P99 is already high and we increase resolution → cost ≈ 1.
        if p99_norm > 0.9 and ca.resolution_delta >= 0 and ca.segmentation_enabled:
            return 1.0
        if p99_norm > 0.8 and ca.resolution_delta > 0:
            return 0.8
        if p99_norm > 0.7:
            return 0.3
        return 0.0

    # ── Training ─────────────────────────────────────────────────────────

    def update_critic(
        self,
        states: torch.Tensor,
        costs: torch.Tensor,
        next_states: torch.Tensor,
    ) -> float:
        """One gradient step on the Lyapunov critic via TD loss.

        L_L = E[(L_φ(s) − (c + γ · L_φ(s')))²]

        Parameters
        ----------
        states:
            ``(B, 11)`` current states.
        costs:
            ``(B,)`` instantaneous constraint costs.
        next_states:
            ``(B, 11)`` successor states.

        Returns
        -------
        float
            Critic loss value.
        """
        self.critic.train()

        L_s = self.critic(states).squeeze(-1)
        with torch.no_grad():
            L_s_next = self.critic(next_states).squeeze(-1)
        target = costs + self._gamma * L_s_next

        loss = nn.functional.mse_loss(L_s, target)

        self._critic_optim.zero_grad()
        loss.backward()
        self._critic_optim.step()

        return loss.item()

    def update_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> float:
        """One gradient step on the transition model via MSE.

        Parameters
        ----------
        states:
            ``(B, 11)`` current states.
        actions:
            ``(B,)`` integer actions.
        next_states:
            ``(B, 11)`` observed successor states.

        Returns
        -------
        float
            Transition model loss.
        """
        self.transition.train()

        oh = torch.zeros(
            actions.shape[0], _NUM_ACTIONS, device=self._device
        )
        oh.scatter_(1, actions.unsqueeze(1).long(), 1.0)
        predicted = self.transition(states, oh)
        loss = nn.functional.mse_loss(predicted, next_states)

        self._trans_optim.zero_grad()
        loss.backward()
        self._trans_optim.step()

        return loss.item()

    # ── Lyapunov value query ─────────────────────────────────────────────

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> float:
        """Return L_φ(s) for a single state."""
        self.critic.eval()
        s = state.to(self._device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        val = self.critic(s).item()
        return val if np.isfinite(val) else 0.0

    # ── Checkpoint I/O ───────────────────────────────────────────────────

    def save(self, checkpoint_dir: Path) -> None:
        """Save critic and transition model weights."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.critic.state_dict(),
            checkpoint_dir / "lyapunov_critic.pt",
        )
        torch.save(
            self.transition.state_dict(),
            checkpoint_dir / "transition_model.pt",
        )
        logger.info("Saved Lyapunov checkpoint to %s", checkpoint_dir)

    def load(self, checkpoint_dir: Path) -> None:
        """Load critic and transition model weights."""
        critic_path = checkpoint_dir / "lyapunov_critic.pt"
        trans_path = checkpoint_dir / "transition_model.pt"

        if critic_path.exists():
            self.critic.load_state_dict(
                torch.load(critic_path, map_location=self._device, weights_only=True)
            )
            logger.info("Loaded Lyapunov critic from %s", critic_path)

        if trans_path.exists():
            self.transition.load_state_dict(
                torch.load(trans_path, map_location=self._device, weights_only=True)
            )
            logger.info("Loaded transition model from %s", trans_path)

        self._nan_detected = False
