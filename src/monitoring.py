"""
src/monitoring.py
─────────────────
Telemetry collection, Parquet logging with rotation, sliding-window
statistics, and GPU monitoring via NVML.

Public API
──────────
    TelemetryLogger   — append-only Parquet writer with rotation
    GPUMonitor        — cached NVML queries for utilisation / temperature / memory
    MetricsWindow     — deque-based sliding window for P50/P95/P99
"""

from __future__ import annotations

import bisect
import logging
import os
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.state_features import TelemetryRecord

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_OUTPUT_DIR: str = "traces/"
_DEFAULT_ROTATION_FRAMES: int = 100_000
_DEFAULT_WINDOW_SIZE: int = 50
_NVML_QUERY_INTERVAL: int = 5  # query every N frames (aligned with controller)


# ── Sliding-window statistics ─────────────────────────────────────────────────

class MetricsWindow:
    """Deque-based sliding window that maintains sorted order for fast quantiles.

    Parameters
    ----------
    window_size:
        Maximum number of samples retained (default 50).
    """

    def __init__(self, window_size: int = _DEFAULT_WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._values: Deque[float] = deque(maxlen=window_size)
        self._sorted: List[float] = []

    def push(self, value: float) -> None:
        """Add *value* to the window, evicting the oldest if full."""
        if len(self._values) == self._window_size:
            oldest = self._values[0]
            idx = bisect.bisect_left(self._sorted, oldest)
            if idx < len(self._sorted) and self._sorted[idx] == oldest:
                self._sorted.pop(idx)

        self._values.append(value)
        bisect.insort(self._sorted, value)

    @property
    def mean(self) -> float:
        if not self._sorted:
            return 0.0
        return sum(self._sorted) / len(self._sorted)

    def percentile(self, p: float) -> float:
        """Return the *p*-th percentile (0-100) of the current window."""
        n = len(self._sorted)
        if n == 0:
            return 0.0
        idx = int(p / 100.0 * (n - 1))
        idx = max(0, min(idx, n - 1))
        return self._sorted[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def count(self) -> int:
        return len(self._values)

    def __len__(self) -> int:
        return len(self._values)


# ── GPU Monitor ───────────────────────────────────────────────────────────────

class GPUMonitor:
    """Cached NVML queries for GPU utilisation, temperature, and memory.

    NVML is only polled every *query_interval* frames.  Between polls the
    last-known values are returned.  If NVML initialisation fails (e.g. no
    GPU), the monitor degrades gracefully and returns zeros.
    """

    def __init__(self, query_interval: int = _NVML_QUERY_INTERVAL) -> None:
        self._query_interval = max(1, query_interval)
        self._nvml_available: bool = False
        self._handle: Any = None

        # Cached values
        self._util_percent: float = 0.0
        self._temp_celsius: float = 0.0
        self._mem_used_mb: float = 0.0

        self._frame_counter: int = 0
        self._last_query_ns: int = 0
        self._stale: bool = True

        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
            logger.info("NVML initialised for GPU 0")
        except Exception:
            self._nvml_available = False
            logger.info("NVML unavailable; GPU metrics will be zero")

    def update(self, force: bool = False) -> None:
        """Refresh cached GPU metrics if due (or *force* is ``True``)."""
        self._frame_counter += 1
        if not self._nvml_available:
            return
        if not force and (self._frame_counter % self._query_interval != 0):
            return

        t0 = time.perf_counter_ns()
        try:
            import pynvml
            util_info = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            self._util_percent = float(util_info.gpu)
            self._temp_celsius = float(
                pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU,
                )
            )
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._mem_used_mb = mem_info.used / (1024 * 1024)
            self._stale = False
        except Exception:
            self._stale = True
            logger.debug("NVML query failed; using last-known values")

        query_ns = time.perf_counter_ns() - t0
        if query_ns > 1_000_000:  # > 1 ms
            self._stale = True
            logger.debug("NVML query took %.2f ms — marking stale", query_ns / 1e6)
        self._last_query_ns = query_ns

    @property
    def utilization(self) -> float:
        return self._util_percent

    @property
    def temperature(self) -> float:
        return self._temp_celsius

    @property
    def memory_used_mb(self) -> float:
        return self._mem_used_mb

    @property
    def is_stale(self) -> bool:
        return self._stale

    def shutdown(self) -> None:
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_available = False


# ── Telemetry Logger ──────────────────────────────────────────────────────────

class TelemetryLogger:
    """Append-only Parquet writer with automatic file rotation.

    Parameters
    ----------
    output_dir:
        Directory for Parquet files. Created if missing.
    rotation_frames:
        Start a new file after this many records.
    window_size:
        Sliding-window length for latency statistics.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        rotation_frames: Optional[int] = None,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ) -> None:
        self._output_dir = Path(
            output_dir or os.environ.get("TELEMETRY_DIR", _DEFAULT_OUTPUT_DIR)
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._rotation_frames = rotation_frames or int(
            os.environ.get("TELEMETRY_ROTATION", str(_DEFAULT_ROTATION_FRAMES))
        )

        # In-memory buffer for the current Parquet segment.
        self._buffer: List[Dict[str, Any]] = []
        self._total_records: int = 0
        self._file_index: int = 0

        # Sliding window for real-time statistics.
        self._latency_window = MetricsWindow(window_size)

        # Dashboard consumption buffer (last N records).
        self._dashboard_buffer: Deque[Dict[str, Any]] = deque(maxlen=window_size)

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, record: TelemetryRecord) -> None:
        """Append a single ``TelemetryRecord`` and rotate if needed."""
        row = asdict(record)
        self._buffer.append(row)
        self._dashboard_buffer.append(row)
        self._latency_window.push(record.latency_ms)
        self._total_records += 1

        if len(self._buffer) >= self._rotation_frames:
            self.flush()

    def flush(self) -> Optional[Path]:
        """Write the in-memory buffer to a Parquet file and reset."""
        if not self._buffer:
            return None

        df = pd.DataFrame(self._buffer)
        filename = self._output_dir / f"telemetry_{self._file_index:05d}.parquet"
        try:
            df.to_parquet(filename, engine="pyarrow", index=False)
            logger.info(
                "Wrote %d records to %s", len(self._buffer), filename,
            )
        except Exception:
            logger.warning("Telemetry write failed for %s — dropping buffer", filename)
            # Recovery priority: never block inference for telemetry.

        self._buffer.clear()
        self._file_index += 1
        return filename

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def latency_mean(self) -> float:
        return self._latency_window.mean

    @property
    def latency_p50(self) -> float:
        return self._latency_window.p50

    @property
    def latency_p95(self) -> float:
        return self._latency_window.p95

    @property
    def latency_p99(self) -> float:
        return self._latency_window.p99

    @property
    def total_records(self) -> int:
        return self._total_records

    def get_dashboard_data(self) -> List[Dict[str, Any]]:
        """Return the most recent records for dashboard consumption."""
        return list(self._dashboard_buffer)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Flush remaining buffer and release resources."""
        self.flush()
        logger.info("TelemetryLogger shutdown — total records: %d", self._total_records)
