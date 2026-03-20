"""
src/trt_common.py
─────────────────
Shared TensorRT utilities used by both the detection and segmentation engines.

Public API
──────────
    TRTLogger
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import tensorrt as trt
except ImportError:
    trt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class TRTLogger(trt.ILogger if trt else object):  # type: ignore[misc]
    """Bridge TensorRT log messages into Python logging."""

    _LEVEL_MAP = (
        {
            trt.Logger.INTERNAL_ERROR: logging.CRITICAL,
            trt.Logger.ERROR: logging.ERROR,
            trt.Logger.WARNING: logging.WARNING,
            trt.Logger.INFO: logging.INFO,
            trt.Logger.VERBOSE: logging.DEBUG,
        }
        if trt
        else {}
    )

    def __init__(self) -> None:
        if trt:
            super().__init__()

    def log(self, severity: Any, msg: str) -> None:  # noqa: A003
        py_level = self._LEVEL_MAP.get(severity, logging.DEBUG)
        logger.log(py_level, "[TensorRT] %s", msg)
