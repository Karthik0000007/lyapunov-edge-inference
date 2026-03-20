"""
tests/test_state_features.py
─────────────────────────────
Unit tests for ControllerState.to_tensor() normalization.
"""

import pytest
import torch

from src.state_features import ControllerState


def _make_state(**overrides) -> ControllerState:
    """Create a ControllerState with sensible defaults, overridden as needed."""
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


class TestToTensorShape:
    """Verify to_tensor() returns correct shape and dtype."""

    def test_shape_is_11(self):
        t = _make_state().to_tensor()
        assert t.shape == (11,)

    def test_dtype_is_float32(self):
        t = _make_state().to_tensor()
        assert t.dtype == torch.float32


class TestNormalizationBounds:
    """All features must be in [0, 1] for boundary inputs."""

    def test_all_minimum_values(self):
        s = ControllerState(
            last_latency_ms=5.0,
            mean_latency_ms=5.0,
            p99_latency_ms=5.0,
            detection_count=0,
            mean_confidence=0.0,
            defect_area_ratio=0.0,
            resolution_index=0,
            threshold_index=0,
            segmentation_enabled=0,
            gpu_utilization=0.0,
            gpu_temperature_norm=30.0,
        )
        t = s.to_tensor()
        assert t.shape == (11,)
        for i in range(11):
            assert 0.0 <= t[i].item() <= 1.0, f"Feature {i} out of [0,1]: {t[i].item()}"

    def test_all_maximum_values(self):
        s = ControllerState(
            last_latency_ms=100.0,
            mean_latency_ms=100.0,
            p99_latency_ms=100.0,
            detection_count=50,
            mean_confidence=1.0,
            defect_area_ratio=1.0,
            resolution_index=2,
            threshold_index=2,
            segmentation_enabled=1,
            gpu_utilization=100.0,
            gpu_temperature_norm=100.0,
        )
        t = s.to_tensor()
        for i in range(11):
            assert 0.0 <= t[i].item() <= 1.0, f"Feature {i} out of [0,1]: {t[i].item()}"

    def test_all_min_are_zero(self):
        """All-minimum inputs should map to exactly 0.0."""
        s = ControllerState(
            last_latency_ms=5.0,
            mean_latency_ms=5.0,
            p99_latency_ms=5.0,
            detection_count=0,
            mean_confidence=0.0,
            defect_area_ratio=0.0,
            resolution_index=0,
            threshold_index=0,
            segmentation_enabled=0,
            gpu_utilization=0.0,
            gpu_temperature_norm=30.0,
        )
        t = s.to_tensor()
        for i in range(11):
            assert t[i].item() == pytest.approx(
                0.0
            ), f"Feature {i} should be 0.0 at minimum, got {t[i].item()}"

    def test_all_max_are_one(self):
        """All-maximum inputs should map to exactly 1.0."""
        s = ControllerState(
            last_latency_ms=100.0,
            mean_latency_ms=100.0,
            p99_latency_ms=100.0,
            detection_count=50,
            mean_confidence=1.0,
            defect_area_ratio=1.0,
            resolution_index=2,
            threshold_index=2,
            segmentation_enabled=1,
            gpu_utilization=100.0,
            gpu_temperature_norm=100.0,
        )
        t = s.to_tensor()
        for i in range(11):
            assert t[i].item() == pytest.approx(
                1.0
            ), f"Feature {i} should be 1.0 at maximum, got {t[i].item()}"


class TestNormalizationMapping:
    """Verify exact numerical values for known inputs."""

    def test_latency_50ms_normalizes_correctly(self):
        """50 ms latency → (50 - 5) / (100 - 5) = 45/95 ≈ 0.47368."""
        s = _make_state(last_latency_ms=50.0)
        t = s.to_tensor()
        expected = (50.0 - 5.0) / (100.0 - 5.0)
        assert t[0].item() == pytest.approx(expected, abs=1e-5)

    def test_detection_count_25_normalizes_to_half(self):
        """25 detections → 25/50 = 0.5."""
        s = _make_state(detection_count=25)
        t = s.to_tensor()
        assert t[3].item() == pytest.approx(0.5, abs=1e-5)

    def test_mean_confidence_passthrough(self):
        """mean_confidence already in [0,1] → clamped passthrough."""
        s = _make_state(mean_confidence=0.85)
        t = s.to_tensor()
        assert t[4].item() == pytest.approx(0.85, abs=1e-5)

    def test_defect_area_ratio_passthrough(self):
        s = _make_state(defect_area_ratio=0.3)
        t = s.to_tensor()
        assert t[5].item() == pytest.approx(0.3, abs=1e-5)

    def test_resolution_index_mapping(self):
        """resolution_index / 2.0: 0→0.0, 1→0.5, 2→1.0."""
        for idx, expected in [(0, 0.0), (1, 0.5), (2, 1.0)]:
            s = _make_state(resolution_index=idx)
            t = s.to_tensor()
            assert t[6].item() == pytest.approx(expected, abs=1e-5)

    def test_threshold_index_mapping(self):
        """threshold_index / 2.0: 0→0.0, 1→0.5, 2→1.0."""
        for idx, expected in [(0, 0.0), (1, 0.5), (2, 1.0)]:
            s = _make_state(threshold_index=idx)
            t = s.to_tensor()
            assert t[7].item() == pytest.approx(expected, abs=1e-5)

    def test_segmentation_binary(self):
        """segmentation_enabled: 0→0.0, 1→1.0."""
        for val, expected in [(0, 0.0), (1, 1.0)]:
            s = _make_state(segmentation_enabled=val)
            t = s.to_tensor()
            assert t[8].item() == pytest.approx(expected, abs=1e-5)

    def test_gpu_utilization_50_percent(self):
        """50% → (50 - 0) / (100 - 0) = 0.5."""
        s = _make_state(gpu_utilization=50.0)
        t = s.to_tensor()
        assert t[9].item() == pytest.approx(0.5, abs=1e-5)

    def test_gpu_temperature_65c(self):
        """65°C → (65 - 30) / (100 - 30) = 35/70 = 0.5."""
        s = _make_state(gpu_temperature_norm=65.0)
        t = s.to_tensor()
        assert t[10].item() == pytest.approx(0.5, abs=1e-5)


class TestExtremeValues:
    """Test with extreme / out-of-range values (clamped to [0,1])."""

    def test_all_zero_state(self):
        """All-zero raw values should clamp to [0, 1]."""
        s = ControllerState(
            last_latency_ms=0.0,
            mean_latency_ms=0.0,
            p99_latency_ms=0.0,
            detection_count=0,
            mean_confidence=0.0,
            defect_area_ratio=0.0,
            resolution_index=0,
            threshold_index=0,
            segmentation_enabled=0,
            gpu_utilization=0.0,
            gpu_temperature_norm=0.0,
        )
        t = s.to_tensor()
        for i in range(11):
            assert 0.0 <= t[i].item() <= 1.0

    def test_very_high_values_clamped(self):
        """Values far above max should clamp to 1.0."""
        s = ControllerState(
            last_latency_ms=500.0,
            mean_latency_ms=500.0,
            p99_latency_ms=500.0,
            detection_count=200,
            mean_confidence=2.0,
            defect_area_ratio=5.0,
            resolution_index=2,
            threshold_index=2,
            segmentation_enabled=1,
            gpu_utilization=200.0,
            gpu_temperature_norm=200.0,
        )
        t = s.to_tensor()
        for i in range(11):
            assert 0.0 <= t[i].item() <= 1.0
