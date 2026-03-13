"""
src/telemetry.py
────────────────
Per-stage timing instrumentation using nanosecond-resolution counters.

Public API
──────────
    StageTiming   — context manager for measuring a single pipeline stage
    FrameTimer    — aggregates all stage timings into a timing dict
"""

from __future__ import annotations

import time
from typing import Dict, Optional


class StageTiming:
    """Context manager that records wall-clock nanoseconds for a code block.

    Usage::

        with StageTiming("preprocess") as t:
            result = preprocess(frame)
        print(t.elapsed_ms)   # e.g. 1.23

    Attributes
    ----------
    name:
        Human-readable label for the stage (e.g. ``"preprocess"``).
    elapsed_ns:
        Elapsed time in nanoseconds (set after ``__exit__``).
    elapsed_ms:
        Elapsed time in milliseconds (``elapsed_ns / 1e6``).
    """

    __slots__ = ("name", "elapsed_ns", "_t0")

    def __init__(self, name: str) -> None:
        self.name = name
        self.elapsed_ns: int = 0
        self._t0: int = 0

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "StageTiming":
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ns = time.perf_counter_ns() - self._t0

    # ── Computed property ─────────────────────────────────────────────────────

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_ns / 1_000_000.0

    def __repr__(self) -> str:
        return f"StageTiming({self.name!r}, {self.elapsed_ms:.3f} ms)"


class FrameTimer:
    """Accumulates per-stage ``StageTiming`` measurements for a single frame.

    Usage::

        timer = FrameTimer()
        with timer.stage("preprocess"):
            preprocess(frame)
        with timer.stage("detect"):
            detect(frame)
        with timer.stage("segment"):
            segment(frame)
        with timer.stage("postprocess"):
            post(frame)

        print(timer.total_ms)           # sum of all stages
        print(timer.as_dict())          # {'preprocess': 1.2, 'detect': 8.3, ...}
    """

    def __init__(self) -> None:
        self._stages: Dict[str, StageTiming] = {}
        self._order: list[str] = []
        self._frame_start_ns: int = time.perf_counter_ns()

    def stage(self, name: str) -> StageTiming:
        """Return a new (or existing) ``StageTiming`` context manager for *name*."""
        st = StageTiming(name)
        self._stages[name] = st
        if name not in self._order:
            self._order.append(name)
        return st

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_ms(self, name: str) -> float:
        """Return the elapsed time (ms) for stage *name*, or ``0.0``."""
        st = self._stages.get(name)
        return st.elapsed_ms if st is not None else 0.0

    @property
    def total_ms(self) -> float:
        """Sum of all recorded stage durations (ms)."""
        return sum(s.elapsed_ms for s in self._stages.values())

    @property
    def total_ns(self) -> int:
        """Sum of all recorded stage durations (ns)."""
        return sum(s.elapsed_ns for s in self._stages.values())

    def as_dict(self) -> Dict[str, float]:
        """Return ``{stage_name: elapsed_ms}`` in insertion order."""
        return {name: self._stages[name].elapsed_ms for name in self._order}

    @property
    def preprocess_ms(self) -> float:
        return self.get_ms("preprocess")

    @property
    def detect_ms(self) -> float:
        return self.get_ms("detect")

    @property
    def segment_ms(self) -> float:
        return self.get_ms("segment")

    @property
    def postprocess_ms(self) -> float:
        return self.get_ms("postprocess")

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}={self._stages[n].elapsed_ms:.2f}ms" for n in self._order)
        return f"FrameTimer({parts})"
