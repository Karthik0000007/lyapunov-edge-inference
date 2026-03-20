"""
tests/test_reward.py
────────────────────
Unit tests for reward computation functions.
"""

import pytest

from src.reward import action_l1_distance, compute_reward, decode_action


class TestDecodeAction:
    """Tests for decode_action() function."""

    def test_decode_action_boundaries(self):
        """decode_action handles boundary values correctly."""
        # Action 0: (res=-1, thr=-1, seg=0)
        res, thr, seg = decode_action(0)
        assert (res, thr, seg) == (-1, -1, 0)

        # Action 17: (res=+1, thr=+1, seg=1)
        res, thr, seg = decode_action(17)
        assert (res, thr, seg) == (1, 1, 1)

    def test_decode_all_18_actions(self):
        """All 18 actions decode to valid factored components."""
        seen = set()
        for a in range(18):
            res, thr, seg = decode_action(a)

            assert res in (-1, 0, 1), f"Action {a}: res={res} invalid"
            assert thr in (-1, 0, 1), f"Action {a}: thr={thr} invalid"
            assert seg in (0, 1), f"Action {a}: seg={seg} invalid"

            # Each action should be unique
            key = (res, thr, seg)
            seen.add(key)

        assert len(seen) == 18, "Not all 18 actions produce unique factored actions"

    def test_decode_action_center_action(self):
        """Center action (no change, no seg) decodes correctly."""
        # (res=0, thr=0, seg=0) = 1*6 + 1*2 + 0 = 8
        res, thr, seg = decode_action(8)
        assert (res, thr, seg) == (0, 0, 0)

    @pytest.mark.parametrize("action_idx", range(18))
    def test_decode_inverse_of_encoding(self, action_idx):
        """Decoding is invertiable: re-encoding should match original."""
        res, thr, seg = decode_action(action_idx)

        # Re-encode
        res_code = res + 1
        thr_code = thr + 1
        seg_code = seg
        reconstructed = res_code * 6 + thr_code * 2 + seg_code

        assert reconstructed == action_idx


class TestActionL1Distance:
    """Tests for action_l1_distance() function."""

    def test_same_action_zero_distance(self):
        """Same action has L1 distance of 0."""
        for a in range(18):
            assert action_l1_distance(a, a) == 0.0

    def test_symmetric(self):
        """L1 distance is symmetric."""
        for a1 in range(18):
            for a2 in range(18):
                assert action_l1_distance(a1, a2) == action_l1_distance(a2, a1)

    def test_max_distance(self):
        """Maximum L1 distance is 5 (res: 2, thr: 2, seg: 1)."""
        # Action 0: (res=-1, thr=-1, seg=0)
        # Action 17: (res=+1, thr=+1, seg=1)
        dist = action_l1_distance(0, 17)
        # |1-(-1)| + |1-(-1)| + |1-0| = 2 + 2 + 1 = 5
        assert dist == 5.0

    def test_single_component_change(self):
        """Distance for single component change."""
        # Only segmentation changes: action 8 (no seg) vs 9 (seg)
        res1, thr1, seg1 = decode_action(8)
        res2, thr2, seg2 = decode_action(9)
        assert (res1, thr1) == (res2, thr2)  # Same res/thr
        assert seg1 != seg2  # Different seg

        dist = action_l1_distance(8, 9)
        assert dist == 1.0


class TestComputeReward:
    """Tests for compute_reward() function."""

    def test_basic_reward_computation(self):
        """Basic reward formula: R = quality*weight - churn*penalty."""
        # Q = confidence * min(count, n_max) / n_max
        # R = Q - 0.05 * churn
        reward = compute_reward(
            mean_confidence=0.8,
            detection_count=50,
            curr_action=8,  # (0, 0, 0)
            prev_action=8,  # Same action, no churn
            n_max=100,
        )

        # Q = 0.8 * 50/100 = 0.4
        # churn = 0 (same action)
        # R = 0.4 - 0 = 0.4
        assert reward == pytest.approx(0.4)

    def test_reward_with_churn_penalty(self):
        """Reward is reduced by action churn."""
        reward = compute_reward(
            mean_confidence=1.0,
            detection_count=100,
            curr_action=0,  # (res=-1, thr=-1, seg=0)
            prev_action=17,  # (res=+1, thr=+1, seg=1)
            n_max=100,
            churn_penalty=0.05,
        )

        # Q = 1.0 * 100/100 = 1.0
        # churn = 5.0 (max L1 distance)
        # R = 1.0 - 0.05 * 5.0 = 0.75
        assert reward == pytest.approx(0.75)

    def test_first_step_no_churn(self):
        """First step (prev_action=-1) has no churn penalty."""
        reward = compute_reward(
            mean_confidence=0.5,
            detection_count=20,
            curr_action=5,
            prev_action=-1,  # First step
            n_max=100,
        )

        # Q = 0.5 * 20/100 = 0.1
        # churn = 0 (first step)
        # R = 0.1
        assert reward == pytest.approx(0.1)

    def test_reward_clamps_detection_count(self):
        """Detection count is clamped to n_max."""
        reward1 = compute_reward(
            mean_confidence=1.0,
            detection_count=100,
            curr_action=8,
            prev_action=8,
            n_max=100,
        )
        reward2 = compute_reward(
            mean_confidence=1.0,
            detection_count=500,  # Exceeds n_max
            curr_action=8,
            prev_action=8,
            n_max=100,
        )

        # Both should give same Q = 1.0 * 100/100 = 1.0
        assert reward1 == pytest.approx(reward2)

    def test_zero_confidence_zero_quality(self):
        """Zero confidence yields zero quality."""
        reward = compute_reward(
            mean_confidence=0.0,
            detection_count=50,
            curr_action=8,
            prev_action=8,
            n_max=100,
        )

        assert reward == pytest.approx(0.0)

    def test_zero_detections_zero_quality(self):
        """Zero detections yields zero quality."""
        reward = compute_reward(
            mean_confidence=0.9,
            detection_count=0,
            curr_action=8,
            prev_action=8,
            n_max=100,
        )

        assert reward == pytest.approx(0.0)

    def test_quality_weight_scaling(self):
        """Quality weight scales the quality component."""
        base_reward = compute_reward(
            mean_confidence=0.5,
            detection_count=50,
            curr_action=8,
            prev_action=8,
            quality_weight=1.0,
        )

        scaled_reward = compute_reward(
            mean_confidence=0.5,
            detection_count=50,
            curr_action=8,
            prev_action=8,
            quality_weight=2.0,
        )

        # Q = 0.5 * 50/100 = 0.25
        # scaled = 2.0 * 0.25 = 0.5
        assert scaled_reward == pytest.approx(2.0 * base_reward)

    def test_negative_reward_possible(self):
        """Reward can be negative with high churn."""
        reward = compute_reward(
            mean_confidence=0.1,
            detection_count=10,
            curr_action=0,
            prev_action=17,
            n_max=100,
            churn_penalty=0.1,
        )

        # Q = 0.1 * 10/100 = 0.01
        # churn = 5.0
        # R = 0.01 - 0.1 * 5.0 = 0.01 - 0.5 = -0.49
        assert reward < 0.0
        assert reward == pytest.approx(-0.49)


class TestRewardIntegration:
    """Integration tests for reward computation."""

    def test_consistent_with_action_encoding(self):
        """Reward computation uses same action encoding as state_features."""
        from src.state_features import ControllerAction

        # Verify decode_action matches ControllerAction.from_index
        for idx in range(18):
            res, thr, seg = decode_action(idx)
            ca = ControllerAction.from_index(idx)

            assert res == ca.resolution_delta
            assert thr == ca.threshold_delta
            assert bool(seg) == ca.segmentation_enabled

    def test_reward_range_reasonable(self):
        """Reward values are bounded in a reasonable range."""
        import random

        random.seed(42)
        rewards = []
        for _ in range(1000):
            r = compute_reward(
                mean_confidence=random.random(),
                detection_count=random.randint(0, 200),
                curr_action=random.randint(0, 17),
                prev_action=random.randint(-1, 17),
            )
            rewards.append(r)

        # With default weights, rewards should be in [-0.25, 1.0]
        # given max quality=1.0, max churn=5*0.05=0.25
        assert min(rewards) >= -0.3
        assert max(rewards) <= 1.1
