"""
app/dashboard.py
────────────────
Streamlit real-time operator dashboard for the Lyapunov-constrained RL
edge inference pipeline.

Layout
──────
    Header:   Pipeline status badge, FPS counter, GPU temp gauge, drift alert
    Row 1:    P99 gauge, P50 gauge, violation rate %, active config triplet
    Row 2:    Latency time-series (500 frames) + controller action timeline
    Row 3:    Latency CDF, conformal scatter, confidence histogram
    Row 4:    Live annotated video feed
    Sidebar:  Window sizes, pause/resume, manual fallback override

Data Sources
────────────
    - SharedMemory block ``lyapunov_telemetry`` for zero-copy frame transfer
    - Parquet file polling from ``traces/`` as fallback
    - Both sources provide the ``TelemetryRecord`` schema

Launch
──────
    streamlit run app/dashboard.py -- [--traces traces/] [--refresh-ms 1000]
    # Also launched automatically by main.py unless --no-dashboard is passed.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Constants ────────────────────────────────────────────────────────────────

_BUDGET_MS: float = 50.0
_SHM_NAME: str = "lyapunov_telemetry"
_SHM_FRAME_NAME: str = "lyapunov_frame"

_RESOLUTION_MAP = {0: 320, 1: 480, 2: 640}

_COLORS = {
    "green": "#2ca02c",
    "yellow": "#ff7f0e",
    "red": "#d62728",
    "blue": "#1f77b4",
    "purple": "#9467bd",
    "gray": "#7f7f7f",
}


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lyapunov Edge Inference — Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Telemetry columns (mirror TelemetryRecord fields) ───────────────────────

_TELEMETRY_COLS = [
    "frame_id", "timestamp",
    "latency_ms", "latency_preprocess_ms", "latency_detect_ms",
    "latency_segment_ms", "latency_postprocess_ms",
    "detection_count", "mean_confidence", "defect_area_ratio",
    "controller_action", "resolution_active", "segmentation_active",
    "threshold_active",
    "gpu_util_percent", "gpu_temp_celsius", "gpu_memory_used_mb",
    "conformal_upper_bound_ms", "conformal_alpha",
    "ks_p_value", "drift_alert",
    "lyapunov_value", "constraint_cost", "reward",
]


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=2)
def _load_parquet_traces(traces_dir: str, max_rows: int = 5000) -> pd.DataFrame:
    """Load most recent telemetry from Parquet files in *traces_dir*."""
    p = Path(traces_dir)
    if not p.exists():
        return pd.DataFrame(columns=_TELEMETRY_COLS)

    parquet_files = sorted(p.glob("telemetry_*.parquet"))
    if not parquet_files:
        return pd.DataFrame(columns=_TELEMETRY_COLS)

    # Load the most recent file(s) up to max_rows.
    dfs: List[pd.DataFrame] = []
    total = 0
    for f in reversed(parquet_files):
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
            total += len(df)
            if total >= max_rows:
                break
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame(columns=_TELEMETRY_COLS)

    combined = pd.concat(reversed(dfs), ignore_index=True)
    return combined.tail(max_rows).reset_index(drop=True)


def _try_shared_memory() -> Optional[pd.DataFrame]:
    """Attempt to read telemetry from shared memory block."""
    try:
        shm = shared_memory.SharedMemory(name=_SHM_NAME, create=False)
        # Protocol: first 4 bytes = uint32 record count,
        #           followed by packed float64 arrays.
        n_records = struct.unpack_from("I", shm.buf, 0)[0]
        if n_records == 0:
            shm.close()
            return None

        # Each record: 24 float64 values = 192 bytes.
        record_size = len(_TELEMETRY_COLS) * 8
        offset = 4
        data = np.frombuffer(
            shm.buf[offset:offset + n_records * record_size],
            dtype=np.float64,
        ).reshape(n_records, len(_TELEMETRY_COLS))

        df = pd.DataFrame(data, columns=_TELEMETRY_COLS)
        shm.close()
        return df
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _try_shared_memory_frame() -> Optional[np.ndarray]:
    """Attempt to read the latest annotated frame from shared memory."""
    try:
        shm = shared_memory.SharedMemory(name=_SHM_FRAME_NAME, create=False)
        # Protocol: first 12 bytes = uint32 height, uint32 width, uint32 channels.
        h, w, c = struct.unpack_from("III", shm.buf, 0)
        if h == 0 or w == 0:
            shm.close()
            return None
        offset = 12
        frame = np.frombuffer(
            shm.buf[offset:offset + h * w * c],
            dtype=np.uint8,
        ).reshape(h, w, c).copy()
        shm.close()
        return frame
    except FileNotFoundError:
        return None
    except Exception:
        return None


def load_telemetry(traces_dir: str, max_rows: int) -> pd.DataFrame:
    """Load telemetry: try shared memory first, fall back to Parquet."""
    shm_df = _try_shared_memory()
    if shm_df is not None and len(shm_df) > 0:
        return shm_df
    return _load_parquet_traces(traces_dir, max_rows)


# ── Plotly theme helper ──────────────────────────────────────────────────────

def _apply_theme(fig: go.Figure, height: int = 300) -> go.Figure:
    """Apply consistent dark theme for dashboard figures."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=40, r=20, t=35, b=30),
        font=dict(size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,40,1)",
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(80,80,100,0.3)",
        showline=True, linecolor="rgba(80,80,100,0.5)",
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(80,80,100,0.3)",
        showline=True, linecolor="rgba(80,80,100,0.5)",
    )
    return fig


# ── Color coding helpers ─────────────────────────────────────────────────────

def _p99_color(val: float) -> str:
    if val <= 40.0:
        return _COLORS["green"]
    if val <= 50.0:
        return _COLORS["yellow"]
    return _COLORS["red"]


def _viol_color(rate: float) -> str:
    if rate <= 0.01:
        return _COLORS["green"]
    if rate <= 0.05:
        return _COLORS["yellow"]
    return _COLORS["red"]


# ── Sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> Dict[str, Any]:
    """Render sidebar controls and return settings dict."""
    st.sidebar.title("Dashboard Controls")

    traces_dir = st.sidebar.text_input(
        "Traces directory", value="traces/",
        help="Directory containing telemetry Parquet files.",
    )
    max_rows = st.sidebar.slider(
        "Max telemetry rows", 500, 10000, 5000, step=500,
        help="Maximum rows to load from trace files.",
    )
    ts_window = st.sidebar.slider(
        "Time-series window", 100, 2000, 500, step=100,
        help="Number of frames shown in the latency time-series.",
    )
    cdf_window = st.sidebar.slider(
        "CDF window (all-time)", 500, 10000, 5000, step=500,
        help="Number of frames for the latency CDF computation.",
    )
    scatter_window = st.sidebar.slider(
        "Conformal scatter window", 50, 500, 200, step=50,
        help="Number of recent frames for the conformal scatter plot.",
    )
    conf_hist_window = st.sidebar.slider(
        "Confidence hist window", 50, 500, 100, step=50,
        help="Number of recent frames for confidence histogram.",
    )
    refresh_ms = st.sidebar.slider(
        "Refresh interval (ms)", 500, 5000, 1000, step=250,
        help="Dashboard auto-refresh interval.",
    )

    paused = st.sidebar.checkbox("Pause polling", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Override")
    fallback_override = st.sidebar.button(
        "Force Fallback",
        help="Write a fallback signal file that the pipeline can read.",
    )
    if fallback_override:
        _write_fallback_signal(True)
        st.sidebar.success("Fallback signal sent")

    clear_fallback = st.sidebar.button(
        "Clear Fallback",
        help="Remove the fallback signal file.",
    )
    if clear_fallback:
        _write_fallback_signal(False)
        st.sidebar.info("Fallback signal cleared")

    return {
        "traces_dir": traces_dir,
        "max_rows": max_rows,
        "ts_window": ts_window,
        "cdf_window": cdf_window,
        "scatter_window": scatter_window,
        "conf_hist_window": conf_hist_window,
        "refresh_ms": refresh_ms,
        "paused": paused,
    }


def _write_fallback_signal(active: bool) -> None:
    """Write/remove a fallback override signal file."""
    signal_path = Path("traces/.fallback_override")
    if active:
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        signal_path.write_text("1")
    elif signal_path.exists():
        signal_path.unlink()


# ── Header bar ───────────────────────────────────────────────────────────────

def _render_header(df: pd.DataFrame) -> None:
    """Render header with status badge, FPS, GPU temp, drift alert."""
    cols = st.columns([2, 1, 1, 1, 1])

    # Pipeline status.
    if len(df) == 0:
        cols[0].markdown(
            "### :red_circle: **Pipeline Offline**"
        )
    else:
        last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else 0
        age = time.time() - last_ts
        if age < 5.0:
            cols[0].markdown("### :green_circle: **Pipeline Online**")
        elif age < 30.0:
            cols[0].markdown("### :large_yellow_circle: **Pipeline Stale**")
        else:
            cols[0].markdown("### :red_circle: **Pipeline Offline**")

    # FPS.
    if len(df) >= 2 and "timestamp" in df.columns:
        recent = df.tail(50)
        if len(recent) >= 2:
            dt = recent["timestamp"].iloc[-1] - recent["timestamp"].iloc[0]
            fps = (len(recent) - 1) / max(dt, 1e-6)
            cols[1].metric("FPS", f"{fps:.1f}")
        else:
            cols[1].metric("FPS", "—")
    else:
        cols[1].metric("FPS", "—")

    # GPU temperature.
    if len(df) > 0 and "gpu_temp_celsius" in df.columns:
        gpu_temp = float(df["gpu_temp_celsius"].iloc[-1])
        color = "normal" if gpu_temp < 80 else ("off" if gpu_temp < 90 else "inverse")
        cols[2].metric("GPU Temp", f"{gpu_temp:.0f} °C")
    else:
        cols[2].metric("GPU Temp", "— °C")

    # GPU utilization.
    if len(df) > 0 and "gpu_util_percent" in df.columns:
        gpu_util = float(df["gpu_util_percent"].iloc[-1])
        cols[3].metric("GPU Util", f"{gpu_util:.0f}%")
    else:
        cols[3].metric("GPU Util", "—%")

    # Drift alert.
    if len(df) > 0 and "drift_alert" in df.columns:
        drift = bool(df["drift_alert"].iloc[-1])
        if drift:
            cols[4].markdown("### :rotating_light: **Drift Alert**")
        else:
            cols[4].markdown("### :white_check_mark: **No Drift**")
    else:
        cols[4].markdown("### :question: **Drift N/A**")


# ── Row 1: Key metrics ──────────────────────────────────────────────────────

def _render_key_metrics(df: pd.DataFrame) -> None:
    """P99, P50, violation rate, active config."""
    st.subheader("Key Metrics")
    cols = st.columns(4)

    if len(df) == 0:
        for c in cols:
            c.metric("—", "No data")
        return

    latencies = df["latency_ms"].values

    # P99 gauge.
    p99 = float(np.percentile(latencies, 99))
    p99_delta = None
    if len(latencies) > 100:
        p99_prev = float(np.percentile(latencies[:-50], 99))
        p99_delta = f"{p99 - p99_prev:+.1f} ms"
    cols[0].metric(
        "P99 Latency",
        f"{p99:.1f} ms",
        delta=p99_delta,
        delta_color="inverse",
    )

    # P50 gauge.
    p50 = float(np.percentile(latencies, 50))
    cols[1].metric("P50 Latency", f"{p50:.1f} ms")

    # Violation rate.
    if "constraint_cost" in df.columns:
        viol_rate = float(df["constraint_cost"].mean())
        cols[2].metric(
            "Violation Rate",
            f"{viol_rate * 100:.1f}%",
        )
    else:
        viol_count = int(np.sum(latencies > _BUDGET_MS))
        viol_rate = viol_count / len(latencies)
        cols[2].metric("Violation Rate", f"{viol_rate * 100:.1f}%")

    # Active config.
    if "resolution_active" in df.columns:
        res = int(df["resolution_active"].iloc[-1])
        seg = bool(df["segmentation_active"].iloc[-1]) if "segmentation_active" in df.columns else False
        thr = float(df["threshold_active"].iloc[-1]) if "threshold_active" in df.columns else 0.25
        config_str = f"{res}px / thr={thr:.2f} / seg={'ON' if seg else 'OFF'}"
    else:
        config_str = "—"
    cols[3].metric("Active Config", config_str)


# ── Row 2: Time-series charts ───────────────────────────────────────────────

def _render_timeseries(df: pd.DataFrame, window: int) -> None:
    """Latency time-series and controller action timeline."""
    st.subheader("Time Series")

    if len(df) == 0:
        st.info("No telemetry data available.")
        return

    recent = df.tail(window)
    col1, col2 = st.columns(2)

    # Latency time-series.
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(recent))),
            y=recent["latency_ms"].values,
            mode="lines",
            name="Latency",
            line=dict(color=_COLORS["blue"], width=1),
        ))
        fig.add_hline(
            y=_BUDGET_MS, line_dash="dash", line_color=_COLORS["red"],
            annotation_text=f"Budget ({_BUDGET_MS:.0f} ms)",
            annotation_position="top right",
        )
        fig.update_layout(
            title="Frame Latency (ms)",
            xaxis_title="Frame",
            yaxis_title="Latency (ms)",
        )
        _apply_theme(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Controller action timeline.
    with col2:
        if "controller_action" in recent.columns:
            actions = recent["controller_action"].values

            # Decode action into components for color coding.
            seg = actions % 2
            thr_delta = (actions // 2) % 3 - 1
            res_delta = (actions // 6) % 3 - 1

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=("Res Delta", "Thr Delta", "Seg"),
                                vertical_spacing=0.08)
            x = list(range(len(recent)))
            fig.add_trace(go.Scatter(
                x=x, y=res_delta, mode="lines",
                line=dict(color=_COLORS["blue"], width=1), name="Res",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=thr_delta, mode="lines",
                line=dict(color=_COLORS["yellow"], width=1), name="Thr",
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=seg, mode="lines",
                line=dict(color=_COLORS["green"], width=1), name="Seg",
            ), row=3, col=1)
            fig.update_layout(title="Controller Actions", showlegend=False)
            _apply_theme(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No controller action data.")


# ── Row 3: Visualizations ───────────────────────────────────────────────────

def _render_visualizations(
    df: pd.DataFrame,
    cdf_window: int,
    scatter_window: int,
    hist_window: int,
) -> None:
    """Latency CDF, conformal scatter, confidence histogram."""
    st.subheader("Analysis")

    if len(df) == 0:
        st.info("No telemetry data available.")
        return

    col1, col2, col3 = st.columns(3)

    # Latency CDF.
    with col1:
        lat = df["latency_ms"].tail(cdf_window).values
        sorted_lat = np.sort(lat)
        cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)

        p95 = float(np.percentile(lat, 95))
        p99 = float(np.percentile(lat, 99))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted_lat, y=cdf, mode="lines",
            line=dict(color=_COLORS["blue"], width=2), name="CDF",
        ))
        # P95/P99 markers.
        fig.add_vline(x=p95, line_dash="dot", line_color=_COLORS["yellow"],
                      annotation_text=f"P95={p95:.1f}")
        fig.add_vline(x=p99, line_dash="dot", line_color=_COLORS["red"],
                      annotation_text=f"P99={p99:.1f}")
        fig.add_vline(x=_BUDGET_MS, line_dash="dash", line_color=_COLORS["red"])
        fig.update_layout(
            title="Latency CDF",
            xaxis_title="Latency (ms)",
            yaxis_title="CDF",
        )
        _apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Conformal bound vs. actual scatter.
    with col2:
        if "conformal_upper_bound_ms" in df.columns:
            scatter_df = df.tail(scatter_window)
            actual = scatter_df["latency_ms"].values
            bounds = scatter_df["conformal_upper_bound_ms"].values

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=actual, y=bounds, mode="markers",
                marker=dict(size=4, color=_COLORS["blue"], opacity=0.5),
                name="Bound vs Actual",
            ))
            lo = max(0, min(actual.min(), bounds.min()) - 5)
            hi = max(actual.max(), bounds.max()) + 5
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines",
                line=dict(color=_COLORS["gray"], dash="dash"), name="y=x",
            ))
            fig.add_hline(y=_BUDGET_MS, line_dash="dot", line_color=_COLORS["red"])
            fig.update_layout(
                title="Conformal Bound vs Actual",
                xaxis_title="Actual (ms)",
                yaxis_title="Upper Bound (ms)",
            )
            _apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No conformal data.")

    # Confidence histogram.
    with col3:
        if "mean_confidence" in df.columns:
            confs = df["mean_confidence"].tail(hist_window).values

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=confs, nbinsx=30,
                marker_color=_COLORS["purple"], opacity=0.7,
                name="Confidence",
            ))
            fig.update_layout(
                title="Detection Confidence",
                xaxis_title="Mean Confidence",
                yaxis_title="Count",
            )
            _apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No confidence data.")


# ── Row 4: Live video feed ──────────────────────────────────────────────────

def _render_live_feed() -> None:
    """Display latest annotated frame from shared memory or placeholder."""
    st.subheader("Live Feed")

    frame = _try_shared_memory_frame()
    if frame is not None:
        # Convert BGR → RGB for Streamlit display.
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_rgb = frame[:, :, ::-1]
        else:
            frame_rgb = frame
        st.image(frame_rgb, caption="Live Annotated Frame", use_container_width=True)
    else:
        st.info(
            "Live video feed unavailable. Shared memory block "
            f"``{_SHM_FRAME_NAME}`` not found. "
            "The feed activates when the pipeline process is running with "
            "shared memory enabled."
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Dashboard entry point."""

    # Sidebar controls.
    settings = _render_sidebar()

    # Auto-refresh.
    try:
        from streamlit_autorefresh import st_autorefresh
        if not settings["paused"]:
            st_autorefresh(
                interval=settings["refresh_ms"],
                key="dashboard_autorefresh",
            )
    except ImportError:
        # Fallback: manual refresh hint.
        if not settings["paused"]:
            st.caption(
                f"Auto-refresh every {settings['refresh_ms']} ms "
                "(install ``streamlit-autorefresh`` for automatic refresh, "
                "or press **R** to refresh manually)."
            )

    # Load telemetry.
    if settings["paused"]:
        st.warning("Telemetry polling paused.")
        df = pd.DataFrame(columns=_TELEMETRY_COLS)
    else:
        df = load_telemetry(settings["traces_dir"], settings["max_rows"])

    # Render sections.
    _render_header(df)
    st.markdown("---")
    _render_key_metrics(df)
    st.markdown("---")
    _render_timeseries(df, settings["ts_window"])
    st.markdown("---")
    _render_visualizations(
        df,
        cdf_window=settings["cdf_window"],
        scatter_window=settings["scatter_window"],
        hist_window=settings["conf_hist_window"],
    )
    st.markdown("---")
    _render_live_feed()

    # Footer.
    st.markdown("---")
    st.caption(
        f"Lyapunov Edge Inference Dashboard | "
        f"Records loaded: {len(df)} | "
        f"Last refresh: {time.strftime('%H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
