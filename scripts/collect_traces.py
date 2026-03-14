"""
scripts/collect_traces.py
─────────────────────────
Collect training traces by running the inference pipeline with a rule-based
controller in ε-greedy exploration mode.  Outputs Parquet files containing
all TelemetryRecord fields plus exploration metadata.

When the real pipeline hardware (TensorRT engines, camera) is unavailable,
generates synthetic traces from the LatencyEnv simulator.

Usage
─────
    python scripts/collect_traces.py \
        --source synthetic \
        --frames 10000 \
        --epsilon 0.3 \
        --epsilon-decay 0.995 \
        --epsilon-min 0.05 \
        --output-dir traces/exploration
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path for src imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import RuleBasedController
from src.state_features import ControllerAction, ControllerState

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_STATE_DIM: int = 11
_NUM_ACTIONS: int = 18
_LATENCY_BUDGET_MS: float = 50.0

_LATENCY_MIN: float = 5.0
_LATENCY_MAX: float = 100.0
_DET_COUNT_MAX: float = 50.0
_GPU_UTIL_MAX: float = 100.0
_GPU_TEMP_MIN: float = 30.0
_GPU_TEMP_MAX: float = 100.0


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _normalize(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return _clamp01((v - lo) / (hi - lo))


# ── Synthetic trace collector ────────────────────────────────────────────────

def collect_synthetic(
    num_frames: int,
    epsilon_start: float,
    epsilon_decay: float,
    epsilon_min: float,
    seed: int,
) -> pd.DataFrame:
    """Collect traces using the LatencyEnv simulator with ε-greedy exploration.

    Returns a DataFrame with all TelemetryRecord columns plus exploration
    metadata (``explored``, ``epsilon``, ``rule_action``).
    """
    from src.env import LatencyEnv

    env = LatencyEnv(
        trace_path="__synthetic__",  # triggers synthetic generation
        max_steps=num_frames,
        latency_budget_ms=_LATENCY_BUDGET_MS,
        seed=seed,
    )
    controller = RuleBasedController()
    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)
    epsilon = epsilon_start

    records: List[Dict] = []

    # Track pipeline config state for building ControllerState.
    resolution_index = 2
    threshold_index = 0
    segmentation_enabled = 1
    latency_window: List[float] = []
    window_size = 50

    for frame_id in range(num_frames):
        # Build a ControllerState from the observation vector.
        # obs layout: [last_lat, mean_lat, p99_lat, det_count, mean_conf,
        #              area_ratio, res_idx, thr_idx, seg, gpu_util, gpu_temp]
        # Denormalize for ControllerState.
        last_lat = obs[0] * (_LATENCY_MAX - _LATENCY_MIN) + _LATENCY_MIN
        mean_lat = obs[1] * (_LATENCY_MAX - _LATENCY_MIN) + _LATENCY_MIN
        p99_lat = obs[2] * (_LATENCY_MAX - _LATENCY_MIN) + _LATENCY_MIN
        det_count = int(obs[3] * _DET_COUNT_MAX)
        mean_conf = float(obs[4])
        area_ratio = float(obs[5])
        gpu_util = float(obs[9] * _GPU_UTIL_MAX)
        gpu_temp = float(obs[10] * (_GPU_TEMP_MAX - _GPU_TEMP_MIN) + _GPU_TEMP_MIN)

        state = ControllerState(
            last_latency_ms=last_lat,
            mean_latency_ms=mean_lat,
            p99_latency_ms=p99_lat,
            detection_count=det_count,
            mean_confidence=mean_conf,
            defect_area_ratio=area_ratio,
            resolution_index=resolution_index,
            threshold_index=threshold_index,
            segmentation_enabled=segmentation_enabled,
            gpu_utilization=gpu_util,
            gpu_temperature_norm=gpu_temp,
        )

        # Rule-based action.
        rule_ca = controller.select_action(state)
        rule_action = rule_ca.action_index

        # ε-greedy exploration.
        explored = False
        if rng.random() < epsilon:
            action = int(rng.integers(0, _NUM_ACTIONS))
            explored = True
        else:
            action = rule_action

        # Step environment.
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        latency_ms = step_info.get("latency_ms", last_lat)
        constraint_cost = step_info.get("constraint_cost", 0.0)

        # Update tracked config.
        resolution_index = step_info.get("resolution_index", resolution_index)
        threshold_index = step_info.get("threshold_index", threshold_index)
        segmentation_enabled = int(step_info.get("segmentation_enabled", True))

        latency_window.append(latency_ms)
        if len(latency_window) > window_size:
            latency_window = latency_window[-window_size:]

        resolution_map = {0: 320, 1: 480, 2: 640}

        records.append({
            # TelemetryRecord fields.
            "frame_id": frame_id,
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "latency_preprocess_ms": latency_ms * 0.1,
            "latency_detect_ms": latency_ms * 0.5,
            "latency_segment_ms": latency_ms * 0.3 if segmentation_enabled else 0.0,
            "latency_postprocess_ms": latency_ms * 0.1,
            "detection_count": det_count,
            "mean_confidence": mean_conf,
            "defect_area_ratio": area_ratio,
            "controller_action": action,
            "resolution_active": resolution_map.get(resolution_index, 640),
            "segmentation_active": bool(segmentation_enabled),
            "threshold_active": 0.25 + threshold_index * 0.1,
            "gpu_util_percent": gpu_util,
            "gpu_temp_celsius": gpu_temp,
            "gpu_memory_used_mb": 0.0,
            "conformal_upper_bound_ms": 0.0,
            "conformal_alpha": 0.01,
            "ks_p_value": 1.0,
            "drift_alert": False,
            "lyapunov_value": 0.0,
            "constraint_cost": constraint_cost,
            "reward": reward,
            # Exploration metadata.
            "explored": explored,
            "epsilon": epsilon,
            "rule_action": rule_action,
            # State vector (for RL training).
            "state_0": float(obs[0]),
            "state_1": float(obs[1]),
            "state_2": float(obs[2]),
            "state_3": float(obs[3]),
            "state_4": float(obs[4]),
            "state_5": float(obs[5]),
            "state_6": float(obs[6]),
            "state_7": float(obs[7]),
            "state_8": float(obs[8]),
            "state_9": float(obs[9]),
            "state_10": float(obs[10]),
            # Next state vector.
            "next_state_0": float(next_obs[0]),
            "next_state_1": float(next_obs[1]),
            "next_state_2": float(next_obs[2]),
            "next_state_3": float(next_obs[3]),
            "next_state_4": float(next_obs[4]),
            "next_state_5": float(next_obs[5]),
            "next_state_6": float(next_obs[6]),
            "next_state_7": float(next_obs[7]),
            "next_state_8": float(next_obs[8]),
            "next_state_9": float(next_obs[9]),
            "next_state_10": float(next_obs[10]),
        })

        obs = next_obs

        # Decay epsilon.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if terminated or truncated:
            obs, info = env.reset()
            resolution_index = 2
            threshold_index = 0
            segmentation_enabled = 1
            latency_window.clear()

        if (frame_id + 1) % 1000 == 0:
            logger.info(
                "Collected %d/%d frames  ε=%.4f  explored=%d%%",
                frame_id + 1,
                num_frames,
                epsilon,
                int(sum(1 for r in records[-1000:] if r["explored"]) / 10),
            )

    return pd.DataFrame(records)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect ε-greedy exploration traces for RL training.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        choices=["synthetic"],
        help="Trace source.  'synthetic' uses the LatencyEnv simulator.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10000,
        help="Number of frames to collect.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.3,
        help="Initial exploration rate.",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Per-frame multiplicative epsilon decay.",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.05,
        help="Minimum exploration rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("traces/exploration"),
        help="Output directory for Parquet files.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = parse_args(argv)

    logger.info(
        "Collecting %d frames — source=%s  ε₀=%.3f  decay=%.4f  ε_min=%.3f",
        args.frames, args.source, args.epsilon, args.epsilon_decay,
        args.epsilon_min,
    )

    df = collect_synthetic(
        num_frames=args.frames,
        epsilon_start=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "traces.parquet"
    df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Saved %d trace records to %s", len(df), output_path)

    # Summary statistics.
    explore_rate = df["explored"].mean() * 100
    violation_rate = df["constraint_cost"].mean() * 100
    mean_reward = df["reward"].mean()
    logger.info(
        "Summary — exploration=%.1f%%  violations=%.1f%%  mean_reward=%.4f",
        explore_rate, violation_rate, mean_reward,
    )


if __name__ == "__main__":
    main()
