"""
src/camera.py
─────────────
Frame acquisition producer thread with backpressure, pinned memory,
nanosecond timestamping, and graceful shutdown.

Public API
──────────
    CameraCapture
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from src.state_features import Frame

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_SOURCE: Union[int, str] = 0
_DEFAULT_FPS: int = 30
_DEFAULT_QUEUE_SIZE: int = 10
_MAX_RETRIES: int = 3
_RETRY_BACKOFF_S: float = 1.0


def _try_pinned_alloc(height: int, width: int, channels: int = 3) -> Optional[np.ndarray]:
    """Attempt to allocate a pinned-memory buffer via OpenCV CUDA.

    Returns ``None`` when CUDA host-memory is unavailable (e.g. no GPU or
    OpenCV built without CUDA support).
    """
    try:
        host_mem = cv2.cuda.HostMem(
            height, width, cv2.CV_8UC(channels), cv2.cuda.HostMem_PAGE_LOCKED
        )
        return host_mem.createMatHeader()
    except (cv2.error, AttributeError, SystemError):
        return None


class CameraCapture:
    """Daemon producer thread that acquires frames and enqueues them.

    Parameters
    ----------
    source:
        OpenCV VideoCapture source — device index (int) or file path (str).
        Overridden by the ``CAMERA_SOURCE`` environment variable.
    fps:
        Target capture frame rate.  Overridden by ``CAMERA_FPS``.
    queue_size:
        Maximum number of frames buffered before the oldest is dropped
        (*newest-wins* backpressure policy).
    """

    def __init__(
        self,
        source: Union[int, str, None] = None,
        fps: Optional[int] = None,
        queue_size: Optional[int] = None,
    ) -> None:
        env_source = os.environ.get("CAMERA_SOURCE")
        if source is None:
            source = _DEFAULT_SOURCE if env_source is None else env_source
        if env_source is not None:
            try:
                source = int(env_source)
            except ValueError:
                source = env_source

        if fps is None:
            fps = int(os.environ.get("CAMERA_FPS", str(_DEFAULT_FPS)))
        if queue_size is None:
            queue_size = _DEFAULT_QUEUE_SIZE

        # Try to interpret as int (device index) when passed as string digit.
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self._source = source
        self._fps = fps
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=queue_size)
        self._shutdown_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._pinned_buffer: Optional[np.ndarray] = None

        # Counters
        self._frame_id: int = 0
        self._frames_dropped: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the capture device and start the producer thread."""
        self._cap = self._open_capture()
        self._shutdown_event.clear()
        self._thread = threading.Thread(
            target=self._producer_loop,
            name="camera-producer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Camera started  source=%s  fps=%d  queue_size=%d",
            self._source,
            self._fps,
            self._queue.maxsize,
        )

    def stop(self) -> None:
        """Signal the producer thread to stop and wait for it to drain."""
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info(
            "Camera stopped  frames_acquired=%d  frames_dropped=%d",
            self._frame_id,
            self._frames_dropped,
        )

    def get_frame(self, timeout: Optional[float] = None) -> Optional[Frame]:
        """Dequeue the next frame (blocking up to *timeout* seconds)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture:
        """Open ``cv2.VideoCapture`` with retry logic."""
        for attempt in range(1, _MAX_RETRIES + 1):
            cap = cv2.VideoCapture(self._source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FPS, self._fps)
                # Attempt pinned-memory allocation based on first-frame size.
                ret, probe = cap.read()
                if ret and probe is not None:
                    h, w = probe.shape[:2]
                    self._pinned_buffer = _try_pinned_alloc(h, w)
                    if self._pinned_buffer is not None:
                        logger.info("Pinned host memory allocated  (%d x %d)", h, w)
                    else:
                        logger.info("Pinned memory unavailable; using pageable allocation")
                return cap
            cap.release()
            logger.warning(
                "Camera open attempt %d/%d failed for source=%s, retrying in %.1fs",
                attempt,
                _MAX_RETRIES,
                self._source,
                _RETRY_BACKOFF_S,
            )
            time.sleep(_RETRY_BACKOFF_S)

        msg = f"Failed to open camera source={self._source} after {_MAX_RETRIES} attempts"
        logger.critical(msg)
        raise RuntimeError(msg)

    def _producer_loop(self) -> None:
        """Continuously read frames and enqueue them until shutdown."""
        assert self._cap is not None
        interval = 1.0 / self._fps if self._fps > 0 else 0.0

        while not self._shutdown_event.is_set():
            loop_start = time.perf_counter()

            ret, raw = self._cap.read()
            ts_acquired = time.perf_counter_ns()

            if not ret or raw is None:
                # For video files, end-of-stream is expected.
                if isinstance(self._source, str) and not str(self._source).isdigit():
                    logger.info("End of video file reached")
                    break
                # For live devices, treat as transient error.
                logger.warning("Frame decode failed, skipping  frame_id=%d", self._frame_id)
                self._frame_id += 1
                self._frames_dropped += 1
                continue

            # Copy into pinned buffer if available.
            if self._pinned_buffer is not None and self._pinned_buffer.shape == raw.shape:
                np.copyto(self._pinned_buffer, raw)
                image = self._pinned_buffer.copy()
            else:
                image = raw

            frame = Frame(
                frame_id=self._frame_id,
                timestamp_acquired=float(ts_acquired),
                timestamp_preprocessed=0.0,
                timestamp_detected=0.0,
                timestamp_segmented=0.0,
                timestamp_completed=0.0,
                raw_image=image,
                preprocessed=image,  # placeholder until preprocessing
                active_resolution=(image.shape[0], image.shape[1]),
                detections=[],
                masks=None,
                controller_action=8,  # default: keep-all, no-op
                latency_ms=0.0,
                stage2_executed=False,
            )

            # Newest-wins: drop oldest if queue full.
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._frames_dropped += 1
                    logger.debug("Queue full — dropped oldest frame")
                except queue.Empty:
                    pass

            self._queue.put(frame)
            self._frame_id += 1

            # Pace capture to target FPS.
            elapsed = time.perf_counter() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "CameraCapture":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
