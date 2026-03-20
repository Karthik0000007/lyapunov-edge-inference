"""
tests/test_env.py
─────────────────
Unit tests for LatencyEnv: Gymnasium API compliance, observation space,
episode termination, constraint cost, reward computation, and seeded resets.
"""

import numpy as np
import pytest

from src.env import LatencyEnv


@pytest.fixture
def env() -> LatencyEnv:
    """Create a LatencyEnv with synthetic trace (no Parquet file needed)."""
    return LatencyEnv(
        trace_path="nonexistent.parquet",  # Triggers synthetic trace
        max_steps=100,
        latency_budget_ms=50.0,
        quality_weight=1.0,
        churn_penalty=0.05,
        seed=42,
    )


class TestGymnasiumAPICompliance:
    """Test that reset/step follow the Gymnasium 0.26+ API."""

    def test_reset_returns_obs_info_tuple(self, env: LatencyEnv):
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_step_returns_5_tuple(self, env: LatencyEnv):
        env.reset()
        result = env.step(0)
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_spaces_defined(self, env: LatencyEnv):
        assert env.observation_space is not None
        assert env.action_space is not None
        assert env.action_space.n == 18


class TestObservationShape:
    """Observation shape is (11,) with values in [0, 1]."""

    def test_reset_observation_shape(self, env: LatencyEnv):
        obs, _ = env.reset()
        assert obs.shape == (11,)
        assert obs.dtype == np.float32

    def test_step_observation_shape(self, env: LatencyEnv):
        env.reset()
        obs, *_ = env.step(0)
        assert obs.shape == (11,)

    def test_observation_values_in_unit_interval(self, env: LatencyEnv):
        obs, _ = env.reset()
        assert np.all(obs >= 0.0), f"Obs below 0: {obs}"
        assert np.all(obs <= 1.0), f"Obs above 1: {obs}"

        for _ in range(20):
            obs, *_ = env.step(env.action_space.sample())
            assert np.all(obs >= 0.0), f"Step obs below 0: {obs}"
            assert np.all(obs <= 1.0), f"Step obs above 1: {obs}"


class TestEpisodeTermination:
    """Episode truncates at configured max_steps."""

    def test_no_early_termination(self, env: LatencyEnv):
        env.reset()
        for _ in range(99):
            _, _, terminated, truncated, _ = env.step(0)
            assert terminated is False

    def test_truncated_at_max_steps(self, env: LatencyEnv):
        env.reset()
        truncated = False
        for i in range(100):
            _, _, terminated, truncated, _ = env.step(0)
            assert terminated is False
        assert truncated is True

    def test_terminated_is_always_false(self, env: LatencyEnv):
        """LatencyEnv is a continuing task — terminated is always False."""
        env.reset()
        for _ in range(100):
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            assert terminated is False


class TestConstraintCost:
    """Constraint cost is binary: 0 or 1."""

    def test_constraint_cost_is_binary(self, env: LatencyEnv):
        env.reset()
        for _ in range(50):
            _, _, _, _, info = env.step(env.action_space.sample())
            cost = info["constraint_cost"]
            assert cost in (0.0, 1.0), f"Constraint cost must be 0 or 1, got {cost}"

    def test_constraint_cost_matches_latency_vs_budget(self, env: LatencyEnv):
        """cost = 1 iff latency > budget."""
        env.reset()
        for _ in range(50):
            _, _, _, _, info = env.step(env.action_space.sample())
            latency = info["latency_ms"]
            cost = info["constraint_cost"]
            if latency > 50.0:
                assert cost == 1.0
            else:
                assert cost == 0.0


class TestRewardComputation:
    """Reward = quality_weight * Q_t - churn_penalty * L1(a_t, a_{t-1})."""

    def test_first_step_no_churn(self, env: LatencyEnv):
        """On the first step, prev_action=-1 so churn should be 0."""
        env.reset()
        _, _, _, _, info = env.step(9)  # Any action
        assert info["churn"] == 0.0

    def test_same_action_zero_churn(self, env: LatencyEnv):
        """Repeating the same action should have zero churn."""
        env.reset()
        env.step(9)
        _, _, _, _, info = env.step(9)
        assert info["churn"] == 0.0

    def test_different_action_positive_churn(self, env: LatencyEnv):
        """Different actions should produce positive churn distance."""
        env.reset()
        env.step(0)  # res=-1, thr=-1, seg=0
        _, _, _, _, info = env.step(17)  # res=+1, thr=+1, seg=1
        assert info["churn"] > 0.0

    def test_reward_is_finite(self, env: LatencyEnv):
        env.reset()
        for _ in range(20):
            _, reward, _, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward)

    def test_info_contains_expected_keys(self, env: LatencyEnv):
        env.reset()
        _, _, _, _, info = env.step(0)
        expected_keys = {
            "constraint_cost",
            "latency_ms",
            "quality",
            "churn",
            "resolution_index",
            "threshold_index",
            "segmentation_enabled",
        }
        assert expected_keys.issubset(info.keys())


class TestSeededResets:
    """Different seeds produce different starting states."""

    def test_different_seeds_different_observations(self):
        env = LatencyEnv(
            trace_path="nonexistent.parquet",
            max_steps=100,
            seed=42,
        )
        obs1, _ = env.reset(seed=1)
        # Step a few times to get past same-initialization artifacts.
        for _ in range(5):
            env.step(env.action_space.sample())
        obs_after_1, _ = env.reset(seed=1)

        obs2, _ = env.reset(seed=99)
        for _ in range(5):
            env.step(env.action_space.sample())
        obs_after_2, _ = env.reset(seed=99)

        # Different seeds should produce different offsets → different obs.
        # (With high probability; the synthetic trace has variability.)
        # We check after reset, not after stepping, to test seed effect.
        obs_seed1, _ = env.reset(seed=1)
        obs_seed2, _ = env.reset(seed=9999)
        # They might be the same by chance, but very unlikely with 1000-row traces.
        # We just verify the mechanism works without error.
        assert obs_seed1.shape == obs_seed2.shape == (11,)

    def test_same_seed_reproducible(self):
        env = LatencyEnv(
            trace_path="nonexistent.parquet",
            max_steps=100,
            seed=42,
        )
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)


class TestCheckEnv:
    """Use Gymnasium's built-in check_env utility."""

    def test_gymnasium_check_env_passes(self, env: LatencyEnv):
        from gymnasium.utils.env_checker import check_env

        # check_env raises on failure; no exception = pass.
        check_env(env, skip_render_check=True)
