"""
src/detection.py
────────────────
TensorRT YOLOv8-Nano detection engine with multi-resolution support,
pre-allocated CUDA buffers, async stream execution, and NMS post-processing.

Public API
──────────
    DetectionEngine
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pycuda.autoinit  # noqa: F401 — initialises CUDA context on import
    import pycuda.driver as cuda
except ImportError:
    cuda = None  # type: ignore[assignment]

try:
    import tensorrt as trt
except ImportError:
    trt = None  # type: ignore[assignment]

from src.state_features import Detection

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_SUPPORTED_RESOLUTIONS: List[int] = [320, 480, 640]
_NUM_CLASSES: int = 80  # YOLOv8 default; overridden from engine output shape
_YOLO_INPUT_CHANNELS: int = 3


# ── TensorRT Logger ─────────────────────────────────────────────────────────

class _TRTLogger(trt.ILogger if trt else object):  # type: ignore[misc]
    """Bridge TensorRT log messages into Python logging."""

    _LEVEL_MAP = {
        trt.Logger.INTERNAL_ERROR: logging.CRITICAL,
        trt.Logger.ERROR: logging.ERROR,
        trt.Logger.WARNING: logging.WARNING,
        trt.Logger.INFO: logging.INFO,
        trt.Logger.VERBOSE: logging.DEBUG,
    } if trt else {}

    def __init__(self) -> None:
        if trt:
            super().__init__()

    def log(self, severity: Any, msg: str) -> None:  # noqa: A003
        py_level = self._LEVEL_MAP.get(severity, logging.DEBUG)
        logger.log(py_level, "[TensorRT] %s", msg)


# ── Per-resolution engine wrapper ────────────────────────────────────────────

class _SingleEngine:
    """Holds a deserialised TensorRT engine with pre-allocated I/O buffers."""

    __slots__ = (
        "resolution",
        "engine",
        "context",
        "stream",
        "h_input",
        "d_input",
        "h_output",
        "d_output",
        "input_shape",
        "output_shape",
    )

    def __init__(
        self,
        engine_path: Path,
        resolution: int,
        trt_logger: _TRTLogger,
    ) -> None:
        self.resolution = resolution

        # ── Deserialise engine ────────────────────────────────────────────
        raw = engine_path.read_bytes()
        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(raw)
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialise TensorRT engine: {engine_path}"
            )
        self.context = self.engine.create_execution_context()

        # ── Identify I/O tensor shapes ────────────────────────────────────
        self.input_shape: Tuple[int, ...] = tuple(
            self.engine.get_tensor_shape(self.engine.get_tensor_name(0))
        )
        self.output_shape: Tuple[int, ...] = tuple(
            self.engine.get_tensor_shape(self.engine.get_tensor_name(1))
        )

        # ── CUDA stream ──────────────────────────────────────────────────
        self.stream = cuda.Stream()

        # ── Pre-allocate pinned host + device buffers ─────────────────────
        input_nbytes = int(np.prod(self.input_shape) * np.float32().itemsize)
        output_nbytes = int(np.prod(self.output_shape) * np.float32().itemsize)

        self.h_input = cuda.pagelocked_empty(
            int(np.prod(self.input_shape)), dtype=np.float32
        )
        self.d_input = cuda.mem_alloc(input_nbytes)

        self.h_output = cuda.pagelocked_empty(
            int(np.prod(self.output_shape)), dtype=np.float32
        )
        self.d_output = cuda.mem_alloc(output_nbytes)

        logger.info(
            "Loaded detection engine: %s  input=%s  output=%s",
            engine_path.name,
            self.input_shape,
            self.output_shape,
        )

    def free(self) -> None:
        """Release device memory (called on engine shutdown)."""
        try:
            self.d_input.free()
            self.d_output.free()
        except Exception:  # noqa: BLE001
            pass


# ── DetectionEngine ──────────────────────────────────────────────────────────

class DetectionEngine:
    """Multi-resolution YOLOv8-Nano TensorRT inference engine.

    Parameters
    ----------
    config:
        ``detection`` section from ``config/pipeline.yaml``.  Expected keys::

            resolutions: [320, 480, 640]
            default_resolution: 640
            confidence_base: 0.25
            confidence_steps: [0.0, 0.1, 0.2]
            iou_threshold: 0.45
            max_detections: 100
            engine_dir: models/detection/
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        if trt is None or cuda is None:
            raise RuntimeError(
                "TensorRT and PyCUDA are required for DetectionEngine. "
                "Install them with: pip install tensorrt pycuda"
            )

        self._resolutions: List[int] = config.get("resolutions", _SUPPORTED_RESOLUTIONS)
        self._default_resolution: int = config.get("default_resolution", 640)
        self._conf_base: float = config.get("confidence_base", 0.25)
        self._conf_steps: List[float] = config.get("confidence_steps", [0.0, 0.1, 0.2])
        self._iou_threshold: float = config.get("iou_threshold", 0.45)
        self._max_detections: int = config.get("max_detections", 100)
        engine_dir = Path(config.get("engine_dir", "models/detection/"))

        # ── Build resolution → index mapping ──────────────────────────────
        self._res_to_idx: Dict[int, int] = {
            r: i for i, r in enumerate(self._resolutions)
        }

        # ── Shared TRT logger ─────────────────────────────────────────────
        self._trt_logger = _TRTLogger()

        # ── Load one engine per resolution ────────────────────────────────
        self._engines: List[_SingleEngine] = []
        for res in self._resolutions:
            path = engine_dir / f"yolov8n_{res}.engine"
            if not path.exists():
                raise FileNotFoundError(
                    f"Detection engine file not found: {path}"
                )
            try:
                self._engines.append(
                    _SingleEngine(path, res, self._trt_logger)
                )
            except cuda.MemoryError:
                logger.critical(
                    "CUDA OOM while loading detection engine %s – "
                    "attempting recovery by freeing previously loaded engines",
                    path,
                )
                self._free_engines()
                raise

        # ── Active engine index ───────────────────────────────────────────
        self._active_idx: int = self._res_to_idx.get(
            self._default_resolution, len(self._resolutions) - 1
        )

        logger.info(
            "DetectionEngine ready — %d resolutions loaded, active=%d",
            len(self._engines),
            self._resolutions[self._active_idx],
        )

    # ── Resolution switching ─────────────────────────────────────────────

    @property
    def active_resolution(self) -> int:
        """Currently selected engine resolution."""
        return self._resolutions[self._active_idx]

    def set_resolution_index(self, index: int) -> None:
        """Switch the active engine without rebuilding anything.

        Parameters
        ----------
        index:
            Index into ``self._resolutions`` (0-based).
        """
        if not (0 <= index < len(self._engines)):
            logger.warning(
                "Invalid resolution index %d; clamping to [0, %d]",
                index,
                len(self._engines) - 1,
            )
            index = max(0, min(index, len(self._engines) - 1))
        self._active_idx = index

    # ── Inference ────────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        threshold_index: int = 0,
    ) -> List[Detection]:
        """Run YOLOv8-Nano detection on a preprocessed image.

        Parameters
        ----------
        image:
            Preprocessed BGR or BGRE image (H, W, 3|4), ``uint8`` or
            ``float32``.  Will be resized/normalised internally if needed.
        threshold_index:
            Additive confidence offset index into ``confidence_steps``.

        Returns
        -------
        List[Detection]
            Post-NMS detections with normalised bounding boxes.
        """
        eng = self._engines[self._active_idx]
        res = eng.resolution

        # ── Prepare input tensor (NCHW float32, 0-1 normalised) ──────────
        blob = self._preprocess_blob(image, res)

        # ── Async H→D copy ───────────────────────────────────────────────
        try:
            np.copyto(eng.h_input, blob.ravel())
            cuda.memcpy_htod_async(eng.d_input, eng.h_input, eng.stream)

            # ── Execute ──────────────────────────────────────────────────
            eng.context.set_tensor_address(
                eng.engine.get_tensor_name(0), int(eng.d_input)
            )
            eng.context.set_tensor_address(
                eng.engine.get_tensor_name(1), int(eng.d_output)
            )
            eng.context.execute_async_v3(stream_handle=eng.stream.handle)

            # ── Async D→H copy ───────────────────────────────────────────
            cuda.memcpy_dtoh_async(eng.h_output, eng.d_output, eng.stream)
            eng.stream.synchronize()

        except cuda.MemoryError:
            logger.critical(
                "CUDA OOM during detection inference at resolution %d – "
                "discarding frame and attempting recovery",
                res,
            )
            eng.stream.synchronize()
            return []

        # ── Reshape raw output ───────────────────────────────────────────
        raw_output = eng.h_output.reshape(eng.output_shape)

        # ── NaN / Inf guard ──────────────────────────────────────────────
        if not np.isfinite(raw_output).all():
            logger.warning(
                "NaN/Inf detected in detection output; discarding frame"
            )
            return []

        # ── Post-process (NMS) ───────────────────────────────────────────
        conf_threshold = self._conf_base + self._conf_steps[
            min(threshold_index, len(self._conf_steps) - 1)
        ]
        detections = self._postprocess(raw_output, conf_threshold, res)
        return detections

    # ── Preprocessing ────────────────────────────────────────────────────

    @staticmethod
    def _preprocess_blob(image: np.ndarray, resolution: int) -> np.ndarray:
        """Convert a BGR image to an NCHW float32 blob normalised to [0, 1]."""
        # Keep only first 3 channels (drop edge channel if present).
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        # Resize to engine resolution.
        if image.shape[0] != resolution or image.shape[1] != resolution:
            image = cv2.resize(
                image, (resolution, resolution), interpolation=cv2.INTER_LINEAR
            )

        # uint8 → float32 and normalise.
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        # HWC → CHW → NCHW
        blob = np.ascontiguousarray(image.transpose(2, 0, 1)[np.newaxis, ...])
        return blob

    # ── NMS Post-processing ──────────────────────────────────────────────

    def _postprocess(
        self,
        raw: np.ndarray,
        conf_threshold: float,
        resolution: int,
    ) -> List[Detection]:
        """Apply NMS on YOLOv8 raw output and return detections.

        YOLOv8 output shape: (1, num_classes + 4, num_anchors).
        Rows 0-3 are cx, cy, w, h; rows 4+ are per-class scores.
        """
        # Squeeze batch dimension and transpose to (num_anchors, 4 + num_classes).
        output = raw[0].T  # (num_anchors, 4 + C)

        num_classes = output.shape[1] - 4
        boxes_cxcywh = output[:, :4]
        class_scores = output[:, 4:]

        # Best class per anchor.
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Confidence filter.
        mask = confidences >= conf_threshold
        if not np.any(mask):
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # cxcywh → xyxy (in pixel coordinates).
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2.0
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2.0
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2.0
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2.0
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # OpenCV NMS (expects x, y, w, h as integers for NMSBoxes).
        boxes_xywh = np.stack(
            [x1, y1, boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]], axis=1
        ).tolist()
        scores_list = confidences.tolist()

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores_list,
            score_threshold=conf_threshold,
            nms_threshold=self._iou_threshold,
        )
        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()[: self._max_detections]

        # Build Detection list with normalised bounding boxes.
        detections: List[Detection] = []
        for idx in indices:
            bx1 = float(np.clip(boxes_xyxy[idx, 0] / resolution, 0.0, 1.0))
            by1 = float(np.clip(boxes_xyxy[idx, 1] / resolution, 0.0, 1.0))
            bx2 = float(np.clip(boxes_xyxy[idx, 2] / resolution, 0.0, 1.0))
            by2 = float(np.clip(boxes_xyxy[idx, 3] / resolution, 0.0, 1.0))
            detections.append(
                Detection(
                    bbox=(bx1, by1, bx2, by2),
                    class_id=int(class_ids[idx]),
                    confidence=float(confidences[idx]),
                    class_name=f"defect_{class_ids[idx]}",
                )
            )
        return detections

    # ── Lifecycle ────────────────────────────────────────────────────────

    def _free_engines(self) -> None:
        """Release all device memory."""
        for eng in self._engines:
            eng.free()
        self._engines.clear()

    def shutdown(self) -> None:
        """Gracefully release all TensorRT resources."""
        logger.info("Shutting down DetectionEngine")
        self._free_engines()

    def __del__(self) -> None:
        self._free_engines()
