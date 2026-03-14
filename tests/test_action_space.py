"""
tests/test_action_space.py
──────────────────────────
Unit tests for ControllerAction encoding/decoding (3 x 3 x 2 = 18 actions).
"""

import pytest

from src.state_features import ControllerAction


class TestRoundTrip:
    """from_index(to_index(a)) == a for all 18 actions."""

    @pytest.mark.parametrize("idx", range(18))
    def test_round_trip_all_indices(self, idx: int):
        action = ControllerAction.from_index(idx)
        assert action.to_index() == idx

    def test_exhaustive_round_trip(self):
        """Verify all 18 indices round-trip correctly in one sweep."""
        for idx in range(18):
            action = ControllerAction.from_index(idx)
            assert action.to_index() == idx
            assert action.action_index == idx


class TestFactoredDecomposition:
    """Verify the factored decomposition is correct."""

    def test_action_0(self):
        """Index 0: res=-1, thr=-1, seg=False."""
        a = ControllerAction.from_index(0)
        assert a.resolution_delta == -1
        assert a.threshold_delta == -1
        assert a.segmentation_enabled is False

    def test_action_1(self):
        """Index 1: res=-1, thr=-1, seg=True."""
        a = ControllerAction.from_index(1)
        assert a.resolution_delta == -1
        assert a.threshold_delta == -1
        assert a.segmentation_enabled is True

    def test_action_4_most_conservative(self):
        """Index 4: res=-1, thr=+1, seg=False (most conservative)."""
        a = ControllerAction.from_index(4)
        assert a.resolution_delta == -1
        assert a.threshold_delta == 1
        assert a.segmentation_enabled is False

    def test_action_8_noop_seg_off(self):
        """Index 8: res=0, thr=0, seg=False (no-op, seg off)."""
        a = ControllerAction.from_index(8)
        assert a.resolution_delta == 0
        assert a.threshold_delta == 0
        assert a.segmentation_enabled is False

    def test_action_9_noop_seg_on(self):
        """Index 9: res=0, thr=0, seg=True (no-op, seg on)."""
        a = ControllerAction.from_index(9)
        assert a.resolution_delta == 0
        assert a.threshold_delta == 0
        assert a.segmentation_enabled is True

    def test_action_17_max_increase(self):
        """Index 17: res=+1, thr=+1, seg=True (max increase everything)."""
        a = ControllerAction.from_index(17)
        assert a.resolution_delta == 1
        assert a.threshold_delta == 1
        assert a.segmentation_enabled is True

    @pytest.mark.parametrize("idx", range(18))
    def test_resolution_delta_range(self, idx: int):
        a = ControllerAction.from_index(idx)
        assert a.resolution_delta in (-1, 0, 1)

    @pytest.mark.parametrize("idx", range(18))
    def test_threshold_delta_range(self, idx: int):
        a = ControllerAction.from_index(idx)
        assert a.threshold_delta in (-1, 0, 1)

    @pytest.mark.parametrize("idx", range(18))
    def test_segmentation_is_bool(self, idx: int):
        a = ControllerAction.from_index(idx)
        assert isinstance(a.segmentation_enabled, bool)


class TestBoundaryClamping:
    """Test that applying actions at config boundaries doesn't exceed limits."""

    def test_resolution_decrease_at_minimum(self):
        """Applying resolution_delta=-1 when at index 0 should clamp to 0."""
        a = ControllerAction.from_index(0)  # res_delta = -1
        current_res = 0
        new_res = max(0, min(2, current_res + a.resolution_delta))
        assert new_res == 0

    def test_resolution_increase_at_maximum(self):
        """Applying resolution_delta=+1 when at index 2 should clamp to 2."""
        a = ControllerAction.from_index(12)  # res_delta = +1
        current_res = 2
        new_res = max(0, min(2, current_res + a.resolution_delta))
        assert new_res == 2

    def test_threshold_decrease_at_minimum(self):
        a = ControllerAction.from_index(0)  # thr_delta = -1
        current_thr = 0
        new_thr = max(0, min(2, current_thr + a.threshold_delta))
        assert new_thr == 0

    def test_threshold_increase_at_maximum(self):
        a = ControllerAction.from_index(4)  # thr_delta = +1
        current_thr = 2
        new_thr = max(0, min(2, current_thr + a.threshold_delta))
        assert new_thr == 2


class TestInvalidIndices:
    """from_index should raise ValueError for out-of-range indices."""

    def test_negative_index_raises(self):
        with pytest.raises(ValueError):
            ControllerAction.from_index(-1)

    def test_index_18_raises(self):
        with pytest.raises(ValueError):
            ControllerAction.from_index(18)

    def test_index_100_raises(self):
        with pytest.raises(ValueError):
            ControllerAction.from_index(100)


class TestToIndexFromComponents:
    """Verify to_index() encodes components correctly."""

    def test_manual_encoding(self):
        """res_code=0, thr_code=2, seg=0 → 0*6 + 2*2 + 0 = 4."""
        a = ControllerAction(
            action_index=4,
            resolution_delta=-1,
            threshold_delta=1,
            segmentation_enabled=False,
        )
        assert a.to_index() == 4

    def test_all_combinations_unique(self):
        """All 18 from_index values produce unique (res, thr, seg) tuples."""
        seen = set()
        for idx in range(18):
            a = ControllerAction.from_index(idx)
            key = (a.resolution_delta, a.threshold_delta, a.segmentation_enabled)
            assert key not in seen, f"Duplicate decomposition at index {idx}"
            seen.add(key)
        assert len(seen) == 18
