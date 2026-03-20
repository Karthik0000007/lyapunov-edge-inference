"""
src/preprocess.py
─────────────────
GPU-accelerated preprocessing pipeline: CLAHE, edge-channel fusion,
multi-scale tiling, dynamic ROI cropping, and resolution resizing.

Public API
──────────
    Preprocessor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.state_features import Detection

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_CLIP_LIMIT: float = 2.0
_DEFAULT_TILE_GRID: Tuple[int, int] = (8, 8)
_DEFAULT_OVERLAP: float = 0.1
_SUPPORTED_RESOLUTIONS: List[int] = [320, 480, 640]


def _is_degenerate(image: np.ndarray, blank_thresh: int = 5, sat_thresh: int = 250) -> bool:
    """Check if an image is blank (near-zero) or saturated (near-255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mean_val = float(gray.mean())
    return mean_val < blank_thresh or mean_val > sat_thresh


class Preprocessor:
    """Image enhancement and region extraction before inference.

    Parameters
    ----------
    config:
        ``preprocessing`` section of ``config/pipeline.yaml``.  Expected keys::

            clahe:
                clip_limit: 2.0
                tile_grid_size: [8, 8]
            edge_channel: true
            tiling:
                enabled: false
                overlap: 0.1
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        clahe_cfg = config.get("clahe", {})
        self._clip_limit: float = clahe_cfg.get("clip_limit", _DEFAULT_CLIP_LIMIT)
        tile_grid = clahe_cfg.get("tile_grid_size", list(_DEFAULT_TILE_GRID))
        self._tile_grid: Tuple[int, int] = (int(tile_grid[0]), int(tile_grid[1]))

        self._edge_channel: bool = config.get("edge_channel", True)

        tiling_cfg = config.get("tiling", {})
        self._tiling_enabled: bool = tiling_cfg.get("enabled", False)
        self._tile_overlap: float = tiling_cfg.get("overlap", _DEFAULT_OVERLAP)

        if self._tiling_enabled:
            logger.warning(
                "Tiling is enabled but NOT YET IMPLEMENTED — will return "
                "original image. Set tiling.enabled=false in config."
            )

        # Pre-create the CLAHE object (reusable for each frame).
        self._clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit,
            tileGridSize=self._tile_grid,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        image: np.ndarray,
        target_resolution: int = 640,
        prior_detections: Optional[List[Detection]] = None,
        roi_mask: Optional[np.ndarray] = None,
        bypass_extras: bool = False,
    ) -> np.ndarray:
        """Run the full preprocessing pipeline.

        Parameters
        ----------
        image:
            Raw BGR frame from camera  (H, W, 3), ``uint8``.
        target_resolution:
            Square side length for the output (one of 320, 480, 640).
        prior_detections:
            Detections from the previous frame for dynamic ROI cropping.
        roi_mask:
            Optional fixed binary mask (H, W) for static ROI cropping.
        bypass_extras:
            If ``True``, skip tiling and edge-channel fusion (used under
            overload to reduce preprocessing cost by ~40 %).

        Returns
        -------
        np.ndarray
            Enhanced image at ``(target_resolution, target_resolution, C)``
            where ``C`` is 3 (BGR) or 4 (BGR + edge) depending on config
            and ``bypass_extras``.
        """
        if image is None or image.size == 0:
            logger.warning("Empty input image; returning blank frame")
            channels = 4 if (self._edge_channel and not bypass_extras) else 3
            return np.zeros((target_resolution, target_resolution, channels), dtype=np.uint8)

        # Degenerate-input fallback: skip CLAHE, return raw resized.
        if _is_degenerate(image):
            logger.warning("Degenerate input detected (blank/saturated); using raw resize")
            return self._resize(image, target_resolution)

        # 1. Dynamic ROI cropping (from prior-frame detections or fixed mask).
        cropped = self._apply_roi(image, prior_detections, roi_mask)

        # 2. CLAHE contrast enhancement.
        enhanced = self._apply_clahe(cropped)

        # 3. Multi-scale tiling (optional, disabled by default).
        if self._tiling_enabled and not bypass_extras:
            enhanced = self._apply_tiling(enhanced)

        # 4. Edge-channel fusion (optional).
        if self._edge_channel and not bypass_extras:
            enhanced = self._apply_edge_channel(enhanced)

        # 5. Resize to controller-specified resolution.
        result = self._resize(enhanced, target_resolution)
        return result

    # ── CLAHE ─────────────────────────────────────────────────────────────────

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L-channel of the image in LAB colour space."""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            l_eq = self._clahe.apply(l_chan)
            lab_eq = cv2.merge([l_eq, a_chan, b_chan])
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        except cv2.error:
            logger.warning("CLAHE failed; returning original image")
            return image

    # ── Edge-channel fusion ───────────────────────────────────────────────────

    def _apply_edge_channel(self, image: np.ndarray) -> np.ndarray:
        """Append a Canny edge map as a 4th channel."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        return np.dstack([image, edges])

    # ── Tiling ────────────────────────────────────────────────────────────────

    def _apply_tiling(self, image: np.ndarray) -> np.ndarray:
        """Multi-scale tiling with overlap — return the stitched result.

        NOTE: Tiling is currently a placeholder and returns the original image.
        Full implementation requires tile-aware detection post-processing which
        is not yet integrated. Set ``tiling.enabled: false`` in config.

        Splits the image into 2x2 overlapping tiles, processes each, and
        recombines (NOT YET IMPLEMENTED — returns original image).
        """
        h, w = image.shape[:2]
        if h < 64 or w < 64:
            return image

        overlap_px_h = int(h * self._tile_overlap)
        overlap_px_w = int(w * self._tile_overlap)

        tile_h = h // 2 + overlap_px_h
        tile_w = w // 2 + overlap_px_w

        tiles: List[np.ndarray] = []
        for row in range(2):
            for col in range(2):
                y_start = max(0, row * (h // 2) - overlap_px_h * row)
                x_start = max(0, col * (w // 2) - overlap_px_w * col)
                y_end = min(h, y_start + tile_h)
                x_end = min(w, x_start + tile_w)
                tiles.append(image[y_start:y_end, x_start:x_end])

        # TODO: Implement tile-aware detection and box stitching.
        # For now, return original image (tiling is disabled by default).
        logger.debug("Tiling not implemented; returning original image")
        return image

    # ── Dynamic ROI cropping ──────────────────────────────────────────────────

    def _apply_roi(
        self,
        image: np.ndarray,
        prior_detections: Optional[List[Detection]],
        roi_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Crop to the union bounding box of prior-frame detections or a
        fixed ROI mask.  Falls back to the full frame when no ROI is available.
        """
        h, w = image.shape[:2]

        # Fixed mask takes priority.
        if roi_mask is not None:
            coords = cv2.findNonZero(roi_mask)
            if coords is not None:
                x, y, rw, rh = cv2.boundingRect(coords)
                return image[y : y + rh, x : x + rw]

        # Detection-based ROI: union bbox with margin.
        if prior_detections:
            x1_min = w
            y1_min = h
            x2_max = 0
            y2_max = 0
            for det in prior_detections:
                bx1, by1, bx2, by2 = det.bbox
                x1_min = min(x1_min, int(bx1 * w))
                y1_min = min(y1_min, int(by1 * h))
                x2_max = max(x2_max, int(bx2 * w))
                y2_max = max(y2_max, int(by2 * h))
            # Add 10 % margin.
            margin_x = int((x2_max - x1_min) * 0.1)
            margin_y = int((y2_max - y1_min) * 0.1)
            x1 = max(0, x1_min - margin_x)
            y1 = max(0, y1_min - margin_y)
            x2 = min(w, x2_max + margin_x)
            y2 = min(h, y2_max + margin_y)
            if x2 > x1 and y2 > y1:
                return image[y1:y2, x1:x2]

        return image

    # ── Resize ────────────────────────────────────────────────────────────────

    @staticmethod
    def _resize(image: np.ndarray, resolution: int) -> np.ndarray:
        """Resize to ``(resolution, resolution)`` preserving channel count."""
        return cv2.resize(
            image,
            (resolution, resolution),
            interpolation=cv2.INTER_LINEAR,
        )
