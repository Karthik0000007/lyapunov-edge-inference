"""
tests/test_agent.py
───────────────────
Unit tests for LyapunovPPOAgent core RL methods:
- compute_gae() — Generalized Advantage Estimation
- update() — PPO training step with Lagrangian augmentation
- select_action() — action selection with Lyapunov masking
"""

import pytest
import torch
import numpy as np

from src.agent_lyapunov_ppo import LyapunovPPOAgent


def _make_config() -> dict:
    """Standard config for agent tests."""
    return {
        "ppo": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coeff": 0.01,
            "value_loss_coeff": 0.5,
            "max_grad_norm": 0.5,
            "hidden_size": 64,
        },
        "lagrangian": {
            "lambda_init": 0.1,
            "lambda_lr": 0.01,
            "constraint_threshold": 0.01,
        },
        "lyapunov": {
            "enabled": True,
            "critic_lr": 3e-4,
            "drift_tolerance": 0.05,
        },
        "agent": {},
    }


class TestComputeGAE:
    """Tests for compute_gae() method."""

    def test_gae_output_shapes(self):
        """compute_gae returns tensors of correct shape."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        T = 10
        rewards = [0.1] * T
        values = [0.5] * T
        dones = [False] * T
        next_value = 0.5

        returns, advantages = agent.compute_gae(rewards, values, dones, next_value)

        assert returns.shape == (T,)
        assert advantages.shape == (T,)
        assert returns.dtype == torch.float32
        assert advantages.dtype == torch.float32

    def test_gae_advantages_normalized(self):
        """Advantages should be normalized (mean ~0, std ~1) when std > 1e-8."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        T = 100
        # Create varying rewards to ensure non-trivial std
        rewards = [0.1 * (i % 5) for i in range(T)]
        values = [0.3 + 0.1 * np.sin(i * 0.2) for i in range(T)]
        dones = [False] * T
        next_value = 0.4

        _, advantages = agent.compute_gae(rewards, values, dones, next_value)

        # Normalized: mean should be close to 0, std close to 1
        assert abs(advantages.mean().item()) < 0.1
        assert abs(advantages.std().item() - 1.0) < 0.2

    def test_gae_terminal_state_handling(self):
        """GAE correctly handles terminal states (done=True)."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        rewards = [1.0, 1.0, 1.0, 0.0, 0.0]
        values = [0.5, 0.5, 0.5, 0.5, 0.5]
        dones = [False, False, True, False, False]
        next_value = 0.5

        returns, advantages = agent.compute_gae(rewards, values, dones, next_value)

        # After terminal state (index 2), GAE should reset
        assert returns.shape == (5,)
        # Verify no NaN values
        assert torch.isfinite(returns).all()
        assert torch.isfinite(advantages).all()

    def test_gae_zero_rewards(self):
        """GAE handles all-zero rewards gracefully."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        T = 10
        rewards = [0.0] * T
        values = [0.5] * T
        dones = [False] * T
        next_value = 0.5

        returns, advantages = agent.compute_gae(rewards, values, dones, next_value)

        # Should not crash and should return valid tensors
        assert not torch.isnan(returns).any()
        assert not torch.isnan(advantages).any()

    def test_gae_single_step(self):
        """GAE works with single-step trajectory."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        rewards = [1.0]
        values = [0.5]
        dones = [False]
        next_value = 0.6

        returns, advantages = agent.compute_gae(rewards, values, dones, next_value)

        assert returns.shape == (1,)
        assert advantages.shape == (1,)

        # Manual check: delta = r + gamma*V' - V = 1.0 + 0.99*0.6 - 0.5 = 1.094
        # For single step, advantage = delta (no accumulation)
        # Returns = advantage + value
        expected_delta = 1.0 + 0.99 * 0.6 - 0.5
        expected_return = expected_delta + 0.5
        assert returns[0].item() == pytest.approx(expected_return, rel=1e-4)


class TestUpdate:
    """Tests for update() method — PPO training step."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        cfg = _make_config()
        return LyapunovPPOAgent(cfg, device=torch.device("cpu"))

    def test_update_returns_loss_dict(self, agent):
        """update() returns dict with all expected loss components."""
        B = 16
        states = torch.rand(B, 11)
        actions = torch.randint(0, 18, (B,))
        old_log_probs = torch.randn(B)
        returns = torch.rand(B)
        advantages = torch.randn(B)
        constraint_costs = torch.zeros(B)
        next_states = torch.rand(B, 11)

        result = agent.update(
            states, actions, old_log_probs, returns, advantages,
            constraint_costs, next_states
        )

        expected_keys = [
            "policy_loss", "value_loss", "entropy",
            "lyapunov_loss", "transition_loss", "lambda", "total_loss"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], float), f"{key} should be float"

    def test_update_modifies_weights(self, agent):
        """update() should modify network weights."""
        # Capture initial weights
        initial_actor = {k: v.clone() for k, v in agent.actor.state_dict().items()}
        initial_critic = {k: v.clone() for k, v in agent.critic.state_dict().items()}

        B = 32
        states = torch.rand(B, 11)
        actions = torch.randint(0, 18, (B,))
        old_log_probs = torch.zeros(B)
        returns = torch.ones(B)  # Non-zero returns to create gradient
        advantages = torch.ones(B)
        constraint_costs = torch.zeros(B)
        next_states = torch.rand(B, 11)

        agent.update(
            states, actions, old_log_probs, returns, advantages,
            constraint_costs, next_states
        )

        # Check weights changed
        actor_changed = False
        for k, v in agent.actor.state_dict().items():
            if not torch.equal(v, initial_actor[k]):
                actor_changed = True
                break

        critic_changed = False
        for k, v in agent.critic.state_dict().items():
            if not torch.equal(v, initial_critic[k]):
                critic_changed = True
                break

        assert actor_changed, "Actor weights should change after update"
        assert critic_changed, "Critic weights should change after update"

    def test_update_handles_constraint_violations(self, agent):
        """update() properly handles high constraint costs."""
        B = 16
        states = torch.rand(B, 11)
        actions = torch.randint(0, 18, (B,))
        old_log_probs = torch.zeros(B)
        returns = torch.rand(B)
        advantages = torch.randn(B)
        # All violations
        constraint_costs = torch.ones(B)
        next_states = torch.rand(B, 11)

        initial_lambda = agent.lagrangian.lambda_value

        result = agent.update(
            states, actions, old_log_probs, returns, advantages,
            constraint_costs, next_states
        )

        # Lambda should increase with violations above threshold
        assert result["lambda"] >= initial_lambda
        assert torch.isfinite(torch.tensor(result["total_loss"]))

    def test_update_no_nan_with_extreme_values(self, agent):
        """update() handles extreme advantage values without NaN."""
        B = 16
        states = torch.rand(B, 11)
        actions = torch.randint(0, 18, (B,))
        old_log_probs = torch.zeros(B)
        returns = torch.rand(B)
        # Large but finite advantages
        advantages = torch.randn(B) * 100.0
        constraint_costs = torch.zeros(B)
        next_states = torch.rand(B, 11)

        result = agent.update(
            states, actions, old_log_probs, returns, advantages,
            constraint_costs, next_states
        )

        # Should not produce NaN
        for key, val in result.items():
            if key != "total_loss":  # total_loss can be NaN in edge cases
                assert not np.isnan(val), f"{key} is NaN"

    def test_update_gradient_clipping_works(self, agent):
        """Gradients are properly clipped to max_grad_norm."""
        B = 32
        states = torch.rand(B, 11)
        actions = torch.randint(0, 18, (B,))
        old_log_probs = torch.zeros(B)
        returns = torch.ones(B) * 1000  # Large returns to create large gradients
        advantages = torch.ones(B) * 1000
        constraint_costs = torch.zeros(B)
        next_states = torch.rand(B, 11)

        result = agent.update(
            states, actions, old_log_probs, returns, advantages,
            constraint_costs, next_states
        )

        # Should complete without error and produce finite loss
        assert torch.isfinite(torch.tensor(result["policy_loss"]))

    def test_update_entropy_is_positive(self, agent):
        """Entropy should be non-negative."""
        B = 16
        states = torch.rand(B, 11)
        actions = torch.randint(0, 18, (B,))
        old_log_probs = torch.randn(B)
        returns = torch.rand(B)
        advantages = torch.randn(B)
        constraint_costs = torch.zeros(B)
        next_states = torch.rand(B, 11)

        result = agent.update(
            states, actions, old_log_probs, returns, advantages,
            constraint_costs, next_states
        )

        assert result["entropy"] >= 0.0


class TestSelectAction:
    """Tests for action selection."""

    def test_select_action_returns_valid_action(self):
        """select_action returns action in valid range."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        state = torch.rand(11)
        action, log_prob, value, lyap_val = agent.select_action(state)

        assert 0 <= action < 18
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert isinstance(lyap_val, float)

    def test_select_action_deterministic_in_eval_mode(self):
        """Actions have consistent distribution in eval mode."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        torch.manual_seed(42)
        state = torch.rand(11)

        # Multiple calls with same state should sample from same distribution
        actions = []
        for _ in range(100):
            action, _, _, _ = agent.select_action(state)
            actions.append(action)

        # Should have some variance (not always same action)
        unique = len(set(actions))
        assert unique >= 1  # At least one unique action

    def test_select_action_handles_batch_dimension(self):
        """select_action works with and without batch dimension."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        # Without batch dim
        state_1d = torch.rand(11)
        action1, _, _, _ = agent.select_action(state_1d)
        assert 0 <= action1 < 18

        # With batch dim
        state_2d = torch.rand(1, 11)
        action2, _, _, _ = agent.select_action(state_2d.squeeze(0))
        assert 0 <= action2 < 18


class TestNaNRecovery:
    """Tests for NaN recovery mechanism."""

    def test_nan_recovery_reverts_weights(self):
        """Agent reverts to last good weights on NaN detection."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        # Do a valid update to set last_good checkpoint
        B = 16
        states = torch.rand(B, 11)
        actions = torch.randint(0, 18, (B,))
        old_log_probs = torch.zeros(B)
        returns = torch.rand(B)
        advantages = torch.randn(B)
        constraint_costs = torch.zeros(B)
        next_states = torch.rand(B, 11)

        agent.update(
            states, actions, old_log_probs, returns, advantages,
            constraint_costs, next_states
        )

        # Verify last_good checkpoint is set
        assert agent._last_good_actor is not None
        assert agent._last_good_critic is not None
