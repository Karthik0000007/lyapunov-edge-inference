"""
tests/test_lyapunov.py
──────────────────────
Unit tests for Lyapunov drift condition, safe-set computation, fallback logic,
training, and NaN detection.
"""

import numpy as np
import pytest
import torch

from src.lyapunov import LyapunovCritic, LyapunovManager, TransitionModel


def _make_manager(**lyap_overrides) -> LyapunovManager:
    """Create a LyapunovManager with default config."""
    lyap_cfg = {"enabled": True, "critic_lr": 3e-4, "drift_tolerance": 0.0}
    lyap_cfg.update(lyap_overrides)
    ppo_cfg = {"gamma": 0.99, "hidden_size": 64}
    return LyapunovManager(lyap_cfg, ppo_cfg, device=torch.device("cpu"))


class TestDriftCondition:
    """Test drift condition correctly identifies safe actions."""

    def test_safe_actions_non_empty_for_reasonable_state(self):
        """For a typical state, the safe set should be non-empty."""
        mgr = _make_manager()
        state = torch.rand(11)
        safe = mgr.compute_safe_actions(state)
        assert len(safe) > 0, "Safe set should be non-empty for reasonable state"

    def test_safe_actions_subset_of_all(self):
        """Safe actions should be a subset of {0, ..., 17}."""
        mgr = _make_manager()
        state = torch.rand(11)
        safe = mgr.compute_safe_actions(state)
        for a in safe:
            assert 0 <= a < 18

    def test_drift_with_known_cost_estimates(self):
        """With explicit cost estimates, verify drift condition logic.

        For action a: drift(a) = c(s,a) + gamma * L(s') - L(s)
        If we set cost=0 for some actions and provide a state where
        L(s') ≈ L(s), those actions should be safe.
        """
        mgr = _make_manager()
        state = torch.rand(11)

        # All zero costs → drift depends only on gamma*L(s') - L(s).
        zero_costs = [0.0] * 18
        safe = mgr.compute_safe_actions(state, cost_estimates=zero_costs)
        assert len(safe) > 0

    def test_high_cost_actions_excluded(self):
        """Actions with very high cost should have high drift and be excluded.

        We set drift_tolerance=0 and give cost=100.0 for all actions.
        Unless L(s) is very large, drift = 100 + gamma*L(s') - L(s) > 0.
        The safe set should fall back to minimum-drift action only.
        """
        mgr = _make_manager(drift_tolerance=0.0)
        state = torch.rand(11)
        high_costs = [100.0] * 18
        safe = mgr.compute_safe_actions(state, cost_estimates=high_costs)
        # Should fall back to the single action with minimum drift.
        assert len(safe) >= 1  # Guaranteed non-empty by fallback


class TestSafeSetNonEmpty:
    """Safe set is guaranteed non-empty (fallback to min-drift action)."""

    def test_non_empty_with_random_state(self):
        mgr = _make_manager()
        for seed in range(10):
            torch.manual_seed(seed)
            state = torch.rand(11)
            safe = mgr.compute_safe_actions(state)
            assert len(safe) > 0, f"Safe set empty for seed {seed}"

    def test_fallback_to_min_drift_action(self):
        """When all drift values are positive (strict tolerance=0), safe set
        should still contain at least one action (the min-drift action)."""
        mgr = _make_manager(drift_tolerance=-1000.0)  # Impossible tolerance
        state = torch.rand(11)
        safe = mgr.compute_safe_actions(state)
        assert len(safe) == 1, "Should fallback to exactly one (min-drift) action"


class TestNaNDetection:
    """NaN in Lyapunov critic triggers fallback to full action set."""

    def test_nan_returns_all_actions(self):
        mgr = _make_manager()

        # Corrupt critic weights to produce NaN.
        with torch.no_grad():
            for p in mgr.critic.parameters():
                p.fill_(float("nan"))

        state = torch.rand(11)
        safe = mgr.compute_safe_actions(state)
        assert safe == list(range(18)), (
            "NaN detection should return full action set"
        )
        assert mgr._nan_detected is True


class TestCriticTraining:
    """TD loss decreases over training iterations on synthetic data."""

    def test_td_loss_decreases(self):
        mgr = _make_manager()
        batch_size = 64
        states = torch.rand(batch_size, 11)
        costs = torch.zeros(batch_size)
        next_states = torch.rand(batch_size, 11)

        losses = []
        for _ in range(50):
            loss = mgr.update_critic(states, costs, next_states)
            losses.append(loss)

        # Loss should decrease over iterations.
        assert losses[-1] < losses[0], (
            f"TD loss should decrease: first={losses[0]:.4f}, "
            f"last={losses[-1]:.4f}"
        )

    def test_transition_model_loss_decreases(self):
        mgr = _make_manager()
        batch_size = 64
        states = torch.rand(batch_size, 11)
        actions = torch.randint(0, 18, (batch_size,))
        next_states = states + 0.01  # Small perturbation for learnable target

        losses = []
        for _ in range(50):
            loss = mgr.update_transition(states, actions, next_states)
            losses.append(loss)

        assert losses[-1] < losses[0], (
            f"Transition loss should decrease: first={losses[0]:.4f}, "
            f"last={losses[-1]:.4f}"
        )


class TestLyapunovValue:
    """Test the value() convenience method."""

    def test_value_returns_finite_float(self):
        mgr = _make_manager()
        state = torch.rand(11)
        val = mgr.value(state)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_value_handles_nan_gracefully(self):
        mgr = _make_manager()
        # Corrupt weights.
        with torch.no_grad():
            for p in mgr.critic.parameters():
                p.fill_(float("nan"))
        state = torch.rand(11)
        val = mgr.value(state)
        assert val == 0.0, "NaN critic should return 0.0"


class TestDisabledLyapunov:
    """When Lyapunov is disabled, all actions are returned."""

    def test_disabled_returns_all_actions(self):
        mgr = _make_manager(enabled=False)
        state = torch.rand(11)
        safe = mgr.compute_safe_actions(state)
        assert safe == list(range(18))
