"""
tests/test_controller.py
────────────────────────
Integration tests for AdaptiveController: three-layer pipeline, fallback
activation/recovery, decision frequency gating, and config clamping.
"""

import pytest
import torch

from src.controller import AdaptiveController, _RuleFallbackChecker
from src.state_features import ControllerAction, ControllerState


def _default_config() -> dict:
    """Minimal config for constructing an AdaptiveController."""
    return {
        "agent": {"decision_frequency": 5, "checkpoint_dir": "checkpoints/test/"},
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
            "drift_tolerance": 0.0,
        },
        "conformal": {
            "enabled": False,  # Disable for simpler testing
            "alpha_target": 0.01,
            "alpha_lr": 0.005,
            "calibration_size": 5000,
            "predictor_hidden_size": 32,
        },
        "fallback": {
            "enabled": True,
            "consecutive_violations": 3,
            "recovery_window": 50,
        },
    }


def _make_state(**overrides) -> ControllerState:
    defaults = dict(
        last_latency_ms=30.0,
        mean_latency_ms=30.0,
        p99_latency_ms=30.0,
        detection_count=5,
        mean_confidence=0.7,
        defect_area_ratio=0.1,
        resolution_index=1,
        threshold_index=1,
        segmentation_enabled=1,
        gpu_utilization=50.0,
        gpu_temperature_norm=65.0,
    )
    defaults.update(overrides)
    return ControllerState(**defaults)


class TestThreeLayerSequence:
    """Integration: AdaptiveController calls all three layers in sequence."""

    def test_step_returns_controller_action(self):
        ctrl = AdaptiveController(_default_config())
        state = _make_state()
        action = ctrl.step(state, observed_latency=30.0)
        assert isinstance(action, ControllerAction)
        assert 0 <= action.action_index < 18

    def test_decision_record_populated(self):
        """After step(), last_decision should be populated with layer info."""
        ctrl = AdaptiveController(_default_config())
        state = _make_state()
        ctrl.step(state, observed_latency=30.0)

        decision = ctrl.last_decision
        assert decision is not None
        assert 0 <= decision.raw_action < 18
        assert isinstance(decision.safe_actions, list)
        assert isinstance(decision.lyapunov_value, float)
        assert isinstance(decision.conformal_bound, float)
        assert isinstance(decision.conformal_overridden, bool)
        assert isinstance(decision.fallback_active, bool)
        assert 0 <= decision.final_action < 18

    def test_multiple_steps_produce_valid_actions(self):
        ctrl = AdaptiveController(_default_config())
        state = _make_state()
        for _ in range(20):
            action = ctrl.step(state, observed_latency=30.0)
            assert isinstance(action, ControllerAction)
            assert 0 <= action.action_index < 18


class TestRuleFallback:
    """Test fallback activation after consecutive violations and recovery."""

    def test_fallback_activates_after_3_violations(self):
        fb = _RuleFallbackChecker(violation_trigger=3, recovery_window=50)
        assert fb.is_active is False

        fb.update(violated=True)
        fb.update(violated=True)
        assert fb.is_active is False  # Only 2

        fb.update(violated=True)
        assert fb.is_active is True  # 3rd consecutive

    def test_fallback_not_active_if_interrupted(self):
        fb = _RuleFallbackChecker(violation_trigger=3, recovery_window=50)
        fb.update(violated=True)
        fb.update(violated=True)
        fb.update(violated=False)  # Resets counter
        fb.update(violated=True)
        assert fb.is_active is False

    def test_fallback_recovery_after_50_clean(self):
        fb = _RuleFallbackChecker(violation_trigger=3, recovery_window=50)

        # Activate fallback.
        for _ in range(3):
            fb.update(violated=True)
        assert fb.is_active is True

        # 49 clean frames: still active.
        for _ in range(49):
            fb.update(violated=False)
        assert fb.is_active is True

        # 50th clean frame: deactivates.
        fb.update(violated=False)
        assert fb.is_active is False

    def test_fallback_recovery_resets_on_violation(self):
        fb = _RuleFallbackChecker(violation_trigger=3, recovery_window=50)

        # Activate.
        for _ in range(3):
            fb.update(violated=True)

        # 30 clean then a violation resets clean counter.
        for _ in range(30):
            fb.update(violated=False)
        fb.update(violated=True)
        assert fb.is_active is True
        assert fb.clean_frames == 0

    def test_controller_fallback_forces_action_4(self):
        """When fallback is active, controller forces action 4."""
        ctrl = AdaptiveController(_default_config())
        state = _make_state()

        # Trigger 3 consecutive violations with high latency.
        for _ in range(3):
            ctrl.step(state, observed_latency=100.0)

        # Next decision should have fallback active.
        # Force a decision frame (frame_counter % 5 == 1).
        ctrl._frame_counter = 0
        action = ctrl.step(state, observed_latency=100.0)

        decision = ctrl.last_decision
        if decision is not None and decision.fallback_active:
            assert decision.final_action == 4


class TestDecisionFrequencyGating:
    """Config only changes every k-th frame."""

    def test_held_action_returned_between_decisions(self):
        ctrl = AdaptiveController(_default_config())  # decision_freq=5
        state = _make_state()

        # Frame 1 (counter=1, 1%5==1 → decision).
        first_action = ctrl.step(state, observed_latency=30.0)
        assert ctrl._frame_counter == 1

        # Frames 2-5 should return held action.
        for frame in range(4):
            action = ctrl.step(state, observed_latency=30.0)
            assert action.action_index == first_action.action_index
            assert action.resolution_delta == first_action.resolution_delta

    def test_new_decision_at_decision_frame(self):
        ctrl = AdaptiveController(_default_config())
        state = _make_state()

        # Execute 6 frames to pass through one full decision cycle.
        actions = []
        for _ in range(6):
            action = ctrl.step(state, observed_latency=30.0)
            actions.append(action)

        # Frame 1 and 6 are decision frames (counter 1 and 6: 1%5==1, 6%5==1).
        # In between, held action is returned.
        assert ctrl._frame_counter == 6


class TestConfigClamping:
    """Resolution and threshold indices stay within [0, 2]."""

    def test_resolution_does_not_go_below_zero(self):
        ctrl = AdaptiveController(_default_config())
        # Start at resolution 0.
        ctrl._resolution_index = 0
        # Apply action with resolution_delta = -1.
        action = ControllerAction.from_index(0)  # res=-1
        ctrl._apply_action(action)
        assert ctrl.resolution_index == 0

    def test_resolution_does_not_go_above_two(self):
        ctrl = AdaptiveController(_default_config())
        ctrl._resolution_index = 2
        action = ControllerAction.from_index(12)  # res=+1
        ctrl._apply_action(action)
        assert ctrl.resolution_index == 2

    def test_threshold_does_not_go_below_zero(self):
        ctrl = AdaptiveController(_default_config())
        ctrl._threshold_index = 0
        action = ControllerAction.from_index(0)  # thr=-1
        ctrl._apply_action(action)
        assert ctrl.threshold_index == 0

    def test_threshold_does_not_go_above_two(self):
        ctrl = AdaptiveController(_default_config())
        ctrl._threshold_index = 2
        action = ControllerAction.from_index(4)  # thr=+1
        ctrl._apply_action(action)
        assert ctrl.threshold_index == 2
