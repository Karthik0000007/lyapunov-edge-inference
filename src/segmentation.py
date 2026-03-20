"""
src/segmentation.py
───────────────────
TensorRT MobileNetV2-UNet segmentation engine with pre-allocated CUDA buffers,
conditional inference gated by detection confidence, and ROI-based mask mapping.

Public API
──────────
    SegmentationEngine
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except (ImportError, FileNotFoundError):
    cuda = None  # type: ignore[assignment]

try:
    import tensorrt as trt
except ImportError:
    trt = None  # type: ignore[assignment]

from src.state_features import Detection
from src.trt_common import TRTLogger as _TRTLogger

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_SEG_RESOLUTION: int = 256
_SEG_INPUT_CHANNELS: int = 3


# ── SegmentationEngine ───────────────────────────────────────────────────────


class SegmentationEngine:
    """MobileNetV2-UNet TensorRT segmentation engine (256x256).

    Parameters
    ----------
    config:
        ``segmentation`` section from ``config/pipeline.yaml``.  Expected keys::

            enabled: true
            resolution: 256
            min_confidence: 0.5
            engine_path: models/segmentation/unet_fp16.engine
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        if trt is None or cuda is None:
            raise RuntimeError(
                "TensorRT and PyCUDA are required for SegmentationEngine. "
                "Install them with: pip install tensorrt pycuda"
            )

        self._resolution: int = config.get("resolution", _SEG_RESOLUTION)
        self._min_confidence: float = config.get("min_confidence", 0.5)
        engine_path = Path(config.get("engine_path", "models/segmentation/unet_fp16.engine"))

        # ── Controllable enabled flag ─────────────────────────────────────
        self.enabled: bool = config.get("enabled", True)

        # ── Shared TRT logger ─────────────────────────────────────────────
        self._trt_logger = _TRTLogger()

        # ── Deserialise engine ────────────────────────────────────────────
        if not engine_path.exists():
            raise FileNotFoundError(f"Segmentation engine file not found: {engine_path}")
        try:
            raw = engine_path.read_bytes()
            runtime = trt.Runtime(self._trt_logger)
            self._engine = runtime.deserialize_cuda_engine(raw)
            if self._engine is None:
                raise RuntimeError(f"Failed to deserialise TensorRT engine: {engine_path}")
            self._context = self._engine.create_execution_context()
        except cuda.MemoryError:
            logger.critical(
                "CUDA OOM while loading segmentation engine – "
                "disabling segmentation and continuing in detection-only mode"
            )
            self.enabled = False
            self._engine = None
            self._context = None
            return

        # ── I/O tensor shapes ─────────────────────────────────────────────
        self._input_shape: Tuple[int, ...] = tuple(
            self._engine.get_tensor_shape(self._engine.get_tensor_name(0))
        )
        self._output_shape: Tuple[int, ...] = tuple(
            self._engine.get_tensor_shape(self._engine.get_tensor_name(1))
        )

        # ── CUDA stream ──────────────────────────────────────────────────
        self._stream = cuda.Stream()

        # ── Pre-allocate pinned host + device buffers ─────────────────────
        input_nbytes = int(np.prod(self._input_shape) * np.float32().itemsize)
        output_nbytes = int(np.prod(self._output_shape) * np.float32().itemsize)

        self._h_input = cuda.pagelocked_empty(int(np.prod(self._input_shape)), dtype=np.float32)
        self._d_input = cuda.mem_alloc(input_nbytes)

        self._h_output = cuda.pagelocked_empty(int(np.prod(self._output_shape)), dtype=np.float32)
        self._d_output = cuda.mem_alloc(output_nbytes)

        logger.info(
            "SegmentationEngine ready — input=%s  output=%s",
            self._input_shape,
            self._output_shape,
        )

    # ── Inference ────────────────────────────────────────────────────────

    def segment(
        self,
        image: np.ndarray,
        detections: List[Detection],
    ) -> Optional[List[np.ndarray]]:
        """Run conditional segmentation on high-confidence ROI crops.

        Segmentation only executes when:
        1. ``self.enabled`` is ``True``.
        2. At least one detection has confidence >= ``min_confidence``.

        Parameters
        ----------
        image:
            Original BGR frame (H, W, 3), ``uint8``.
        detections:
            Stage 1 detections with normalised bounding boxes.

        Returns
        -------
        Optional[List[np.ndarray]]
            Binary/multi-class masks mapped back to original ROI coordinates,
            one per qualifying detection.  ``None`` if segmentation was
            skipped (disabled or no qualifying detections).
        """
        if not self.enabled or self._engine is None:
            return None

        # Filter detections by confidence.
        qualifying = [d for d in detections if d.confidence >= self._min_confidence]
        if not qualifying:
            return None

        h, w = image.shape[:2]
        masks: List[np.ndarray] = []

        for det in qualifying:
            mask = self._segment_roi(image, det, h, w)
            if mask is not None:
                masks.append(mask)

        return masks if masks else None

    def _segment_roi(
        self,
        image: np.ndarray,
        det: Detection,
        img_h: int,
        img_w: int,
    ) -> Optional[np.ndarray]:
        """Extract ROI, run inference, and map mask back to ROI coordinates."""
        # ── Extract ROI crop ─────────────────────────────────────────────
        x1 = max(0, int(det.bbox[0] * img_w))
        y1 = max(0, int(det.bbox[1] * img_h))
        x2 = min(img_w, int(det.bbox[2] * img_w))
        y2 = min(img_h, int(det.bbox[3] * img_h))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = image[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # ── Resize to engine resolution ──────────────────────────────────
        resized = cv2.resize(
            roi,
            (self._resolution, self._resolution),
            interpolation=cv2.INTER_LINEAR,
        )

        # ── Prepare NCHW float32 blob ────────────────────────────────────
        if resized.ndim == 3 and resized.shape[2] > 3:
            resized = resized[:, :, :3]
        blob = resized.astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])

        # ── Async H→D, execute, D→H ─────────────────────────────────────
        try:
            np.copyto(self._h_input, blob.ravel())
            cuda.memcpy_htod_async(self._d_input, self._h_input, self._stream)

            self._context.set_tensor_address(self._engine.get_tensor_name(0), int(self._d_input))
            self._context.set_tensor_address(self._engine.get_tensor_name(1), int(self._d_output))
            self._context.execute_async_v3(stream_handle=self._stream.handle)

            cuda.memcpy_dtoh_async(self._h_output, self._d_output, self._stream)
            self._stream.synchronize()

        except cuda.MemoryError:
            logger.critical(
                "CUDA OOM during segmentation inference – "
                "disabling segmentation and continuing in detection-only mode"
            )
            self.enabled = False
            return None

        # ── Reshape and produce mask ─────────────────────────────────────
        raw = self._h_output.reshape(self._output_shape)

        # NaN / Inf guard.
        if not np.isfinite(raw).all():
            logger.warning("NaN/Inf detected in segmentation output; skipping ROI")
            return None

        # Handle multi-class (C, H, W) or binary (1, H, W) output.
        if raw.ndim == 4:
            raw = raw[0]  # Remove batch dim → (C, H, W)

        if raw.shape[0] == 1:
            # Binary segmentation: sigmoid → threshold.
            mask_256 = (self._sigmoid(raw[0]) > 0.5).astype(np.uint8)
        else:
            # Multi-class: argmax over channel dim.
            mask_256 = np.argmax(raw, axis=0).astype(np.uint8)

        # ── Map mask back to original ROI spatial dimensions ─────────────
        mask_roi = cv2.resize(
            mask_256,
            (roi_w, roi_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return mask_roi

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    # ── Lifecycle ────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Gracefully release all TensorRT resources."""
        logger.info("Shutting down SegmentationEngine")
        try:
            if hasattr(self, "_d_input"):
                self._d_input.free()
            if hasattr(self, "_d_output"):
                self._d_output.free()
        except Exception:  # noqa: BLE001
            pass
        self._engine = None
        self._context = None

    def __del__(self) -> None:
        self.shutdown()
