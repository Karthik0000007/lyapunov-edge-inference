"""
src/drift.py
────────────
Statistical drift detection for pipeline health monitoring.

* **IntensityDriftDetector** — two-sample KS-test on grayscale histograms
  (current frame vs. reference distribution).
* **ConfidenceDriftDetector** — CUSUM change-point detection on mean
  detection confidence.
* **DriftMonitor** — unified façade exposing alert status and KS p-values.

Public API
──────────
    IntensityDriftDetector
    ConfidenceDriftDetector
    DriftMonitor
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_HISTORY_LEN: int = 20
_DEFAULT_ALERT_THRESHOLD: int = 5  # min frames with p < alpha to trigger
_DEFAULT_ALPHA: float = 0.01  # KS-test significance level
_DEFAULT_NUM_BINS: int = 256  # histogram bins for grayscale


# ── Intensity Drift Detector ─────────────────────────────────────────────────


class IntensityDriftDetector:
    """Two-sample Kolmogorov-Smirnov test on grayscale intensity histograms.

    A reference distribution is captured from the first frame (or explicitly
    set).  Each subsequent frame is compared via the KS statistic.  An alert
    fires when ``p < alpha`` for at least ``alert_threshold`` of the last
    ``history_len`` frames.

    Parameters
    ----------
    alpha:
        Significance level for the KS test (default 0.01).
    history_len:
        Number of recent p-values retained (default 20).
    alert_threshold:
        Minimum p < alpha count within *history_len* to raise an alert
        (default 5).
    num_bins:
        Number of histogram bins for the grayscale distribution (default 256).
    """

    def __init__(
        self,
        alpha: float = _DEFAULT_ALPHA,
        history_len: int = _DEFAULT_HISTORY_LEN,
        alert_threshold: int = _DEFAULT_ALERT_THRESHOLD,
        num_bins: int = _DEFAULT_NUM_BINS,
    ) -> None:
        self._alpha = alpha
        self._history_len = history_len
        self._alert_threshold = alert_threshold
        self._num_bins = num_bins

        self._reference_hist: Optional[np.ndarray] = None
        self._p_value_history: Deque[float] = deque(maxlen=history_len)
        self._last_p_value: float = 1.0

    # ── Public API ────────────────────────────────────────────────────────────

    def set_reference(self, image: np.ndarray) -> None:
        """Capture the reference grayscale histogram from *image*."""
        self._reference_hist = self._compute_histogram(image)

    def update(self, image: np.ndarray) -> float:
        """Compare *image* against the reference and return the KS p-value.

        On the first call, if no reference has been set, the current frame
        becomes the reference and ``p = 1.0`` is returned.
        """
        current_hist = self._compute_histogram(image)

        if self._reference_hist is None:
            self._reference_hist = current_hist
            self._last_p_value = 1.0
            self._p_value_history.append(1.0)
            return 1.0

        # Two-sample KS test on the empirical CDFs derived from histograms.
        _, p_value = stats.ks_2samp(self._reference_hist, current_hist)
        p_value = float(p_value)
        self._last_p_value = p_value
        self._p_value_history.append(p_value)
        return p_value

    @property
    def last_p_value(self) -> float:
        """Most recent KS p-value."""
        return self._last_p_value

    @property
    def alert(self) -> bool:
        """``True`` if drift is detected (p < alpha for >= alert_threshold
        of the last *history_len* frames)."""
        if len(self._p_value_history) == 0:
            return False
        sig_count = sum(1 for p in self._p_value_history if p < self._alpha)
        return sig_count >= self._alert_threshold

    def reset(self) -> None:
        """Clear the reference and p-value history."""
        self._reference_hist = None
        self._p_value_history.clear()
        self._last_p_value = 1.0

    # ── Internals ─────────────────────────────────────────────────────────────

    def _compute_histogram(self, image: np.ndarray) -> np.ndarray:
        """Convert *image* to grayscale and return a normalised histogram."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        hist = cv2.calcHist([gray], [0], None, [self._num_bins], [0, 256])
        hist = hist.flatten().astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist


# ── Confidence Drift Detector (CUSUM) ────────────────────────────────────────


class ConfidenceDriftDetector:
    """CUSUM (Cumulative Sum) change-point detector on mean detection
    confidence.

    The CUSUM algorithm tracks positive and negative deviations from a
    target mean.  When the cumulative sum exceeds a threshold *h* the
    detector signals a change-point (confidence drift).

    Parameters
    ----------
    target_mean:
        Expected baseline mean confidence (default 0.5).
    drift_margin:
        Allowable slack around the target (default 0.05).  Deviations
        smaller than this are not accumulated.
    threshold_h:
        CUSUM decision threshold (default 5.0).  Higher values reduce
        false positives but increase detection delay.
    """

    def __init__(
        self,
        target_mean: float = 0.5,
        drift_margin: float = 0.05,
        threshold_h: float = 5.0,
    ) -> None:
        self._target = target_mean
        self._margin = drift_margin
        self._h = threshold_h

        # CUSUM accumulators (positive / negative direction).
        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._alert: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, mean_confidence: float) -> bool:
        """Feed the latest mean detection confidence and return alert status.

        Returns
        -------
        bool
            ``True`` if CUSUM detects a change-point (drift).
        """
        deviation = mean_confidence - self._target

        # Accumulate positive shifts (mean rising).
        self._s_pos = max(0.0, self._s_pos + deviation - self._margin)
        # Accumulate negative shifts (mean dropping).
        self._s_neg = max(0.0, self._s_neg - deviation - self._margin)

        self._alert = self._s_pos > self._h or self._s_neg > self._h
        return self._alert

    @property
    def alert(self) -> bool:
        """``True`` if confidence drift is currently detected."""
        return self._alert

    @property
    def cusum_pos(self) -> float:
        """Current positive CUSUM accumulator value."""
        return self._s_pos

    @property
    def cusum_neg(self) -> float:
        """Current negative CUSUM accumulator value."""
        return self._s_neg

    def reset(self) -> None:
        """Reset accumulators to zero and clear alert."""
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._alert = False


# ── Unified Drift Monitor ────────────────────────────────────────────────────


class DriftMonitor:
    """Unified façade combining intensity and confidence drift detectors.

    Parameters
    ----------
    ks_alpha:
        Significance level for the KS intensity test (default 0.01).
    ks_history_len:
        Number of recent p-values to track (default 20).
    ks_alert_threshold:
        Number of significant results needed to trigger alert (default 5).
    cusum_target:
        CUSUM target mean confidence (default 0.5).
    cusum_margin:
        CUSUM allowable slack (default 0.05).
    cusum_h:
        CUSUM decision threshold (default 5.0).
    """

    def __init__(
        self,
        ks_alpha: float = _DEFAULT_ALPHA,
        ks_history_len: int = _DEFAULT_HISTORY_LEN,
        ks_alert_threshold: int = _DEFAULT_ALERT_THRESHOLD,
        cusum_target: float = 0.5,
        cusum_margin: float = 0.05,
        cusum_h: float = 5.0,
    ) -> None:
        self._intensity_detector = IntensityDriftDetector(
            alpha=ks_alpha,
            history_len=ks_history_len,
            alert_threshold=ks_alert_threshold,
        )
        self._confidence_detector = ConfidenceDriftDetector(
            target_mean=cusum_target,
            drift_margin=cusum_margin,
            threshold_h=cusum_h,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_reference(self, image: np.ndarray) -> None:
        """Set the reference image for intensity drift detection."""
        self._intensity_detector.set_reference(image)

    def update(
        self,
        image: np.ndarray,
        mean_confidence: float,
    ) -> Tuple[float, bool]:
        """Run both detectors and return ``(ks_p_value, drift_alert)``.

        Parameters
        ----------
        image:
            Current BGR frame for intensity comparison.
        mean_confidence:
            Mean detection confidence for the current frame.

        Returns
        -------
        ks_p_value:
            KS-test p-value from the intensity detector.
        drift_alert:
            ``True`` if *either* detector is currently in alert.
        """
        ks_p = self._intensity_detector.update(image)
        self._confidence_detector.update(mean_confidence)

        alert = self._intensity_detector.alert or self._confidence_detector.alert
        return ks_p, alert

    @property
    def ks_p_value(self) -> float:
        """Most recent KS p-value from the intensity detector."""
        return self._intensity_detector.last_p_value

    @property
    def intensity_alert(self) -> bool:
        """``True`` if the intensity drift detector is in alert."""
        return self._intensity_detector.alert

    @property
    def confidence_alert(self) -> bool:
        """``True`` if the confidence drift detector is in alert."""
        return self._confidence_detector.alert

    @property
    def alert(self) -> bool:
        """``True`` if either detector is in alert."""
        return self._intensity_detector.alert or self._confidence_detector.alert

    def reset(self) -> None:
        """Reset both detectors."""
        self._intensity_detector.reset()
        self._confidence_detector.reset()
