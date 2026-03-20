"""
src/agent_lyapunov_ppo.py
─────────────────────────
Lyapunov-PPO actor-critic agent with:
  - Actor MLP π_θ (11 → 64 → 64 → 18 → softmax)
  - Value critic V_ψ (11 → 64 → 64 → 1)
  - Lyapunov action masking via LyapunovManager
  - Lagrangian dual-variable augmented loss
  - PPO clipped surrogate with GAE (λ=0.95)
  - NaN guard on logits with checkpoint revert

Network parameter counts:
    Actor:     11×64 + 64 + 64×64 + 64 + 64×18 + 18 = 5,970
    Critic:    11×64 + 64 + 64×64 + 64 + 64×1  + 1  = 4,865
    Lyapunov:  (same as critic)                        = 4,865
    Total:                                             ≈ 15,700

Public API
──────────
    LyapunovPPOAgent
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.lagrangian import LagrangianDual
from src.lyapunov import LyapunovManager
from src.torch_compat import torch_load_compat

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_STATE_DIM: int = 11
_NUM_ACTIONS: int = 18


# ── Actor network ───────────────────────────────────────────────────────────


class _Actor(nn.Module):
    """Policy network π_θ(a|s): 11 → hidden → hidden → 18 → softmax."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_STATE_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, _NUM_ACTIONS),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw logits (pre-softmax), shape ``(B, 18)``."""
        return self.net(state)


# ── Value critic ─────────────────────────────────────────────────────────────


class _ValueCritic(nn.Module):
    """Value function V_ψ(s): 11 → 64 → 64 → 1."""

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
        return self.net(state)


# ── LyapunovPPOAgent ────────────────────────────────────────────────────────


class LyapunovPPOAgent:
    """Lyapunov-constrained PPO agent.

    Parameters
    ----------
    config:
        Full controller configuration dict containing at minimum::

            ppo:
                gamma, gae_lambda, clip_epsilon, entropy_coeff,
                value_loss_coeff, max_grad_norm, hidden_size, num_layers
            lagrangian:
                lambda_init, lambda_lr, constraint_threshold
            lyapunov:
                enabled, critic_lr, drift_tolerance
            agent:
                checkpoint_dir
    device:
        Torch device.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device | None = None,
    ) -> None:
        self._device = device or torch.device("cpu")

        ppo_cfg = config.get("ppo", {})
        lag_cfg = config.get("lagrangian", {})
        lyap_cfg = config.get("lyapunov", {})
        agent_cfg = config.get("agent", {})

        self._gamma: float = ppo_cfg.get("gamma", 0.99)
        self._gae_lambda: float = ppo_cfg.get("gae_lambda", 0.95)
        self._clip_eps: float = ppo_cfg.get("clip_epsilon", 0.2)
        self._entropy_coeff: float = ppo_cfg.get("entropy_coeff", 0.01)
        self._value_loss_coeff: float = ppo_cfg.get("value_loss_coeff", 0.5)
        self._max_grad_norm: float = ppo_cfg.get("max_grad_norm", 0.5)
        hidden = ppo_cfg.get("hidden_size", 64)

        self._checkpoint_dir = Path(agent_cfg.get("checkpoint_dir", "checkpoints/ppo_lyapunov/"))

        # ── Networks ─────────────────────────────────────────────────────
        self._actor = _Actor(hidden_size=hidden).to(self._device)
        self._critic = _ValueCritic(hidden_size=hidden).to(self._device)

        # ── Optimizers ───────────────────────────────────────────────────
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=3e-4)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=3e-4)

        # ── Safety layers ────────────────────────────────────────────────
        self._lyapunov = LyapunovManager(lyap_cfg, ppo_cfg, device=self._device)
        self._lagrangian = LagrangianDual(lag_cfg)

        # ── Last-known-good checkpoint for NaN recovery ──────────────────
        self._last_good_actor: Optional[Dict[str, Any]] = None
        self._last_good_critic: Optional[Dict[str, Any]] = None

        # ── Logging ──────────────────────────────────────────────────────
        actor_params = sum(p.numel() for p in self._actor.parameters())
        critic_params = sum(p.numel() for p in self._critic.parameters())
        lyap_params = sum(p.numel() for p in self._lyapunov.critic.parameters())
        total = actor_params + critic_params + lyap_params
        logger.info(
            "LyapunovPPOAgent ready — actor=%d  critic=%d  lyapunov=%d  " "total=%d params",
            actor_params,
            critic_params,
            lyap_params,
            total,
        )

    # ── Action selection ─────────────────────────────────────────────────

    def select_action(self, state: torch.Tensor) -> Tuple[int, float, float, float]:
        """Select an action using the actor with Lyapunov masking.

        Flow:
        1. Actor produces logits for all 18 actions.
        2. Lyapunov critic computes safe set A_safe(s).
        3. Softmax is applied over safe actions only.
        4. Action is sampled from the masked distribution.

        Parameters
        ----------
        state:
            Normalised state tensor ``(11,)``.

        Returns
        -------
        Tuple[int, float, float, float]
            ``(action_index, log_prob, value, lyapunov_value)``.
        """
        self._actor.eval()
        self._critic.eval()

        s = state.to(self._device)
        if s.dim() == 1:
            s = s.unsqueeze(0)

        with torch.no_grad():
            logits = self._actor(s).squeeze(0)  # (18,)

            # NaN guard on logits.
            if not torch.isfinite(logits).all():
                logger.warning(
                    "NaN/Inf in actor logits; reverting to last-known-good "
                    "checkpoint and returning conservative action"
                )
                self._revert_to_last_good()
                return 4, 0.0, 0.0, 0.0  # Most conservative action

            value = self._critic(s).squeeze().item()
            lyap_val = self._lyapunov.value(state)

        # Lyapunov action masking.
        safe_actions = self._lyapunov.compute_safe_actions(state)

        # Mask: set logits of unsafe actions to -inf.
        mask = torch.full((_NUM_ACTIONS,), float("-inf"), device=self._device)
        for a in safe_actions:
            mask[a] = 0.0
        masked_logits = logits + mask

        # Softmax over safe actions → sample.
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), value, lyap_val

    # ── PPO update ───────────────────────────────────────────────────────

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        constraint_costs: torch.Tensor,
        next_states: torch.Tensor,
    ) -> Dict[str, float]:
        """One PPO epoch over a batch of transitions.

        Parameters
        ----------
        states:
            ``(B, 11)`` normalised states.
        actions:
            ``(B,)`` integer actions.
        old_log_probs:
            ``(B,)`` log π_old(a|s).
        returns:
            ``(B,)`` discounted returns.
        advantages:
            ``(B,)`` GAE advantages (normalised).
        constraint_costs:
            ``(B,)`` per-step constraint costs (0. or 1.).
        next_states:
            ``(B, 11)`` successor states.

        Returns
        -------
        Dict[str, float]
            Loss components: ``policy_loss``, ``value_loss``,
            ``entropy``, ``lyapunov_loss``, ``transition_loss``,
            ``lambda``, ``total_loss``.
        """
        # Snapshot for NaN recovery.
        self._last_good_actor = copy.deepcopy(self._actor.state_dict())
        self._last_good_critic = copy.deepcopy(self._critic.state_dict())

        states = states.to(self._device)
        actions = actions.to(self._device)
        old_log_probs = old_log_probs.to(self._device)
        returns = returns.to(self._device)
        advantages = advantages.to(self._device)
        constraint_costs = constraint_costs.to(self._device)
        next_states = next_states.to(self._device)

        # ── Actor loss (PPO clipped surrogate) ───────────────────────────
        self._actor.train()
        logits = self._actor(states)

        # NaN guard.
        if not torch.isfinite(logits).all():
            logger.warning("NaN in actor logits during update; reverting")
            self._revert_to_last_good()
            return {"total_loss": float("nan")}

        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self._clip_eps, 1.0 + self._clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value loss ───────────────────────────────────────────────────
        self._critic.train()
        values = self._critic(states).squeeze(-1)
        value_loss = nn.functional.mse_loss(values, returns)

        # ── Lagrangian augmented loss ────────────────────────────────────
        mean_cost = constraint_costs.mean()
        total_policy_loss = self._lagrangian.augmented_loss(policy_loss, mean_cost)
        total_loss = (
            total_policy_loss + self._value_loss_coeff * value_loss - self._entropy_coeff * entropy
        )

        # ── Backward + clip ──────────────────────────────────────────────
        self._actor_optim.zero_grad()
        self._critic_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), self._max_grad_norm)
        nn.utils.clip_grad_norm_(self._critic.parameters(), self._max_grad_norm)
        self._actor_optim.step()
        self._critic_optim.step()

        # ── Lagrangian dual update ───────────────────────────────────────
        self._lagrangian.update(mean_cost.item())

        # ── Lyapunov critic + transition model update ────────────────────
        lyap_loss = self._lyapunov.update_critic(states, constraint_costs, next_states)
        trans_loss = self._lyapunov.update_transition(states, actions, next_states)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "lyapunov_loss": lyap_loss,
            "transition_loss": trans_loss,
            "lambda": self._lagrangian.lambda_value,
            "total_loss": total_loss.item(),
        }

    # ── GAE computation ──────────────────────────────────────────────────

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        rewards:
            Per-step rewards.
        values:
            Per-step V_ψ(s) estimates.
        dones:
            Per-step done flags (always False for continuing task).
        next_value:
            Bootstrap value V_ψ(s_{T+1}).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(returns, advantages)`` each of shape ``(T,)``.
        """
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self._gamma * next_val * mask - values[t]
            gae = delta + self._gamma * self._gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + torch.tensor(values, dtype=torch.float32)

        # Normalise advantages.
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    # ── NaN recovery ─────────────────────────────────────────────────────

    def _revert_to_last_good(self) -> None:
        """Revert actor and critic to last-known-good checkpoint."""
        if self._last_good_actor is not None:
            self._actor.load_state_dict(self._last_good_actor)
            logger.info("Reverted actor to last-known-good weights")
        if self._last_good_critic is not None:
            self._critic.load_state_dict(self._last_good_critic)
            logger.info("Reverted critic to last-known-good weights")

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def lyapunov(self) -> LyapunovManager:
        """Access the Lyapunov safety layer."""
        return self._lyapunov

    @property
    def lagrangian(self) -> LagrangianDual:
        """Access the Lagrangian dual variable."""
        return self._lagrangian

    @property
    def actor(self) -> nn.Module:
        """Access the actor network."""
        return self._actor

    @property
    def critic(self) -> nn.Module:
        """Access the value critic network."""
        return self._critic

    # ── Checkpoint I/O ───────────────────────────────────────────────────

    def save(self, checkpoint_dir: Optional[Path] = None) -> None:
        """Save all network weights and dual state."""
        d = checkpoint_dir or self._checkpoint_dir
        d.mkdir(parents=True, exist_ok=True)

        torch.save(self._actor.state_dict(), d / "actor.pt")
        torch.save(self._critic.state_dict(), d / "critic.pt")
        torch.save(self._lagrangian.state_dict(), d / "lagrangian.pt")
        self._lyapunov.save(d)
        logger.info("Saved LyapunovPPOAgent to %s", d)

    def load(self, checkpoint_dir: Optional[Path] = None) -> None:
        """Load all network weights and dual state."""
        d = checkpoint_dir or self._checkpoint_dir

        actor_path = d / "actor.pt"
        critic_path = d / "critic.pt"
        lag_path = d / "lagrangian.pt"

        if actor_path.exists():
            self._actor.load_state_dict(torch_load_compat(actor_path, map_location=self._device))
            logger.info("Loaded actor from %s", actor_path)

        if critic_path.exists():
            self._critic.load_state_dict(torch_load_compat(critic_path, map_location=self._device))
            logger.info("Loaded critic from %s", critic_path)

        if lag_path.exists():
            lag_state = torch_load_compat(lag_path, map_location="cpu")
            self._lagrangian.load_state_dict(lag_state)

        self._lyapunov.load(d)
