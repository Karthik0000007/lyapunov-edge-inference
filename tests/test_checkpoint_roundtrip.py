"""
tests/test_checkpoint_roundtrip.py
──────────────────────────────────
Smoke tests for LyapunovPPOAgent checkpoint save/load roundtrip.

Ensures the I/O paths used in main.py (controller.agent.save/load)
work correctly and produce identical state dicts after a roundtrip.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.agent_lyapunov_ppo import LyapunovPPOAgent


def _make_config() -> dict:
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
        "lagrangian": {"lambda_init": 0.1, "lambda_lr": 0.01, "constraint_threshold": 0.01},
        "lyapunov": {"enabled": True, "critic_lr": 3e-4, "drift_tolerance": 0.0},
        "agent": {},
    }


class TestAgentCheckpointRoundtrip:
    """save() then load() produces matching state dicts."""

    def test_roundtrip_state_dict_keys_match(self):
        """All expected checkpoint files are written and loadable."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            agent.save(d)

            expected_files = [
                "actor.pt",
                "critic.pt",
                "lagrangian.pt",
                "lyapunov_critic.pt",
                "transition_model.pt",
            ]
            for fname in expected_files:
                assert (d / fname).exists(), f"Missing checkpoint file: {fname}"

            agent2 = LyapunovPPOAgent(cfg, device=torch.device("cpu"))
            agent2.load(d)

        # Verify actor keys match.
        keys_orig = set(agent.actor.state_dict().keys())
        keys_loaded = set(agent2.actor.state_dict().keys())
        assert keys_orig == keys_loaded

        # Verify critic keys match.
        keys_orig = set(agent.critic.state_dict().keys())
        keys_loaded = set(agent2.critic.state_dict().keys())
        assert keys_orig == keys_loaded

    def test_roundtrip_weights_identical(self):
        """Loaded weights are numerically identical to saved weights."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        # Perturb weights so they're non-default.
        with torch.no_grad():
            for p in agent.actor.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            agent.save(d)

            agent2 = LyapunovPPOAgent(cfg, device=torch.device("cpu"))
            agent2.load(d)

        for (n1, p1), (n2, p2) in zip(
            agent.actor.state_dict().items(),
            agent2.actor.state_dict().items(),
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Mismatch in actor param {n1}"

        for (n1, p1), (n2, p2) in zip(
            agent.critic.state_dict().items(),
            agent2.critic.state_dict().items(),
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Mismatch in critic param {n1}"

    def test_roundtrip_lagrangian_state(self):
        """Lagrangian dual state (lambda) is preserved across save/load."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        # Manually adjust lambda to a non-default value.
        agent.lagrangian.update(0.5)  # Push lambda away from init
        lambda_before = agent.lagrangian.lambda_value

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            agent.save(d)

            agent2 = LyapunovPPOAgent(cfg, device=torch.device("cpu"))
            agent2.load(d)

        assert agent2.lagrangian.lambda_value == pytest.approx(lambda_before)

    def test_load_missing_dir_is_noop(self):
        """Loading from a non-existent directory does not crash."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        state_before = {k: v.clone() for k, v in agent.actor.state_dict().items()}

        agent.load(Path("/nonexistent/path/that/does/not/exist"))

        # Weights should be unchanged.
        for k, v in agent.actor.state_dict().items():
            assert torch.equal(v, state_before[k])

    def test_select_action_after_roundtrip(self):
        """Agent can produce actions after a save/load cycle."""
        cfg = _make_config()
        agent = LyapunovPPOAgent(cfg, device=torch.device("cpu"))

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            agent.save(d)

            agent2 = LyapunovPPOAgent(cfg, device=torch.device("cpu"))
            agent2.load(d)

        state = torch.rand(11)
        action, log_prob, value, lyap_val = agent2.select_action(state)
        assert 0 <= action < 18
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert isinstance(lyap_val, float)
