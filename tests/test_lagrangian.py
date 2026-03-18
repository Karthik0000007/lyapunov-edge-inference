"""
tests/test_lagrangian.py
────────────────────────
Unit tests for LagrangianDual constrained optimization mechanism.
"""

import pytest
import torch

from src.lagrangian import LagrangianDual


def _make_config(
    lambda_init: float = 0.1,
    lambda_lr: float = 0.01,
    constraint_threshold: float = 0.01,
) -> dict:
    """Create Lagrangian config."""
    return {
        "lambda_init": lambda_init,
        "lambda_lr": lambda_lr,
        "constraint_threshold": constraint_threshold,
    }


class TestLagrangianDualInit:
    """Tests for initialization."""

    def test_init_from_config(self):
        """LagrangianDual initializes from config dict."""
        cfg = _make_config(lambda_init=0.5, lambda_lr=0.02, constraint_threshold=0.05)
        dual = LagrangianDual(cfg)

        assert dual.lambda_value == 0.5
        assert dual.threshold == 0.05

    def test_init_default_values(self):
        """LagrangianDual uses defaults for missing config keys."""
        dual = LagrangianDual({})

        assert dual.lambda_value == 0.1  # default
        assert dual.threshold == 0.01    # default


class TestDualUpdate:
    """Tests for dual gradient ascent update."""

    def test_update_increases_lambda_above_threshold(self):
        """Lambda increases when constraint cost > threshold."""
        cfg = _make_config(lambda_init=0.1, lambda_lr=0.1, constraint_threshold=0.01)
        dual = LagrangianDual(cfg)

        initial_lambda = dual.lambda_value
        # Cost = 0.1 > threshold = 0.01
        new_lambda = dual.update(0.1)

        assert new_lambda > initial_lambda
        assert new_lambda == dual.lambda_value

    def test_update_decreases_lambda_below_threshold(self):
        """Lambda decreases when constraint cost < threshold (can hit 0)."""
        cfg = _make_config(lambda_init=0.5, lambda_lr=0.1, constraint_threshold=0.1)
        dual = LagrangianDual(cfg)

        initial_lambda = dual.lambda_value
        # Cost = 0.0 < threshold = 0.1
        new_lambda = dual.update(0.0)

        assert new_lambda < initial_lambda

    def test_update_lambda_never_negative(self):
        """Lambda is clamped to be non-negative."""
        cfg = _make_config(lambda_init=0.01, lambda_lr=1.0, constraint_threshold=0.5)
        dual = LagrangianDual(cfg)

        # Large negative gradient should still result in lambda >= 0
        # delta = lr * (cost - threshold) = 1.0 * (0.0 - 0.5) = -0.5
        # new_lambda = max(0, 0.01 - 0.5) = 0
        new_lambda = dual.update(0.0)

        assert new_lambda >= 0.0

    def test_update_at_threshold_no_change(self):
        """Lambda unchanged when cost equals threshold."""
        cfg = _make_config(lambda_init=0.5, lambda_lr=0.1, constraint_threshold=0.1)
        dual = LagrangianDual(cfg)

        initial_lambda = dual.lambda_value
        new_lambda = dual.update(0.1)  # cost == threshold

        assert new_lambda == pytest.approx(initial_lambda)

    def test_update_convergence_scenario(self):
        """Lambda converges when violations decrease over time."""
        cfg = _make_config(lambda_init=0.1, lambda_lr=0.05, constraint_threshold=0.01)
        dual = LagrangianDual(cfg)

        # Simulate decreasing violations
        lambdas = [dual.lambda_value]
        for violation_rate in [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
            lambdas.append(dual.update(violation_rate))

        # Lambda should first increase then decrease
        assert max(lambdas) > lambdas[0]  # Increases initially
        # After violations drop below threshold, lambda should decrease
        assert lambdas[-1] < max(lambdas)


class TestAugmentedLoss:
    """Tests for augmented loss computation."""

    def test_augmented_loss_formula(self):
        """augmented_loss computes L = policy_loss + lambda * cost."""
        cfg = _make_config(lambda_init=0.5)
        dual = LagrangianDual(cfg)

        policy_loss = torch.tensor(1.0)
        constraint_cost = torch.tensor(0.2)

        aug_loss = dual.augmented_loss(policy_loss, constraint_cost)

        # L = 1.0 + 0.5 * 0.2 = 1.1
        expected = 1.0 + 0.5 * 0.2
        assert aug_loss.item() == pytest.approx(expected)

    def test_augmented_loss_zero_lambda(self):
        """augmented_loss with lambda=0 equals policy_loss."""
        cfg = _make_config(lambda_init=0.0)
        dual = LagrangianDual(cfg)

        policy_loss = torch.tensor(2.5)
        constraint_cost = torch.tensor(1.0)

        aug_loss = dual.augmented_loss(policy_loss, constraint_cost)

        assert aug_loss.item() == pytest.approx(2.5)

    def test_augmented_loss_is_differentiable(self):
        """augmented_loss preserves gradient flow."""
        cfg = _make_config(lambda_init=0.5)
        dual = LagrangianDual(cfg)

        policy_loss = torch.tensor(1.0, requires_grad=True)
        constraint_cost = torch.tensor(0.2, requires_grad=True)

        aug_loss = dual.augmented_loss(policy_loss, constraint_cost)
        aug_loss.backward()

        assert policy_loss.grad is not None
        assert constraint_cost.grad is not None
        assert policy_loss.grad.item() == pytest.approx(1.0)
        assert constraint_cost.grad.item() == pytest.approx(0.5)  # lambda


class TestStateSerialization:
    """Tests for state_dict / load_state_dict."""

    def test_state_dict_contains_all_fields(self):
        """state_dict includes lambda, lr, and threshold."""
        cfg = _make_config(lambda_init=0.3, lambda_lr=0.05, constraint_threshold=0.02)
        dual = LagrangianDual(cfg)
        dual.update(0.1)  # Modify lambda

        state = dual.state_dict()

        assert "lambda" in state
        assert "lr" in state
        assert "threshold" in state
        assert state["lambda"] == dual.lambda_value

    def test_load_state_dict_restores_lambda(self):
        """load_state_dict correctly restores lambda value."""
        cfg = _make_config(lambda_init=0.1)
        dual1 = LagrangianDual(cfg)

        # Modify lambda
        for _ in range(10):
            dual1.update(0.5)

        state = dual1.state_dict()

        # Create new dual and restore
        dual2 = LagrangianDual(cfg)
        dual2.load_state_dict(state)

        assert dual2.lambda_value == pytest.approx(dual1.lambda_value)

    def test_roundtrip_preserves_state(self):
        """Full save/load roundtrip preserves all state."""
        cfg = _make_config(lambda_init=0.2, lambda_lr=0.03, constraint_threshold=0.015)
        dual = LagrangianDual(cfg)

        # Simulate some updates
        for cost in [0.1, 0.05, 0.02]:
            dual.update(cost)

        state = dual.state_dict()

        dual2 = LagrangianDual({})  # Different config
        dual2.load_state_dict(state)

        assert dual2.lambda_value == pytest.approx(dual.lambda_value)
        assert dual2.threshold == pytest.approx(dual.threshold)


class TestEdgeCases:
    """Edge case handling tests."""

    def test_very_large_constraint_cost(self):
        """Handles very large constraint costs gracefully."""
        cfg = _make_config(lambda_init=0.1, lambda_lr=0.01)
        dual = LagrangianDual(cfg)

        # Large cost should just increase lambda proportionally
        new_lambda = dual.update(1000.0)

        assert new_lambda > 0.1
        assert not float('inf') == new_lambda

    def test_very_small_learning_rate(self):
        """Works with very small learning rate."""
        cfg = _make_config(lambda_init=0.1, lambda_lr=1e-8, constraint_threshold=0.01)
        dual = LagrangianDual(cfg)

        initial = dual.lambda_value
        dual.update(1.0)  # Large constraint

        # Should change by tiny amount
        assert dual.lambda_value >= initial
        assert abs(dual.lambda_value - initial) < 1e-5

    def test_zero_constraint_cost(self):
        """Handles zero constraint cost."""
        cfg = _make_config(lambda_init=0.5, lambda_lr=0.1, constraint_threshold=0.1)
        dual = LagrangianDual(cfg)

        new_lambda = dual.update(0.0)

        # Should decrease since 0 < threshold
        assert new_lambda < 0.5
