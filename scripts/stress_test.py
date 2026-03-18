"""
scripts/stress_test.py
──────────────────────
Six stress-test scenarios for the Lyapunov-PPO agent.

Scenarios:
    1. Steady state          — normal operating conditions
    2. Defect burst          — 80 % defect rate for 200 frames
    3. GPU contention        — simulated 30 % GPU load increase
    4. Thermal throttle      — sustained thermal ramp
    5. Distribution shift    — brightness/noise injection at frame 500
    6. Combined stress       — all stressors overlaid simultaneously

Usage
─────
    python scripts/stress_test.py \
        --checkpoint checkpoints/ppo_lyapunov \
        --traces data/telemetry.parquet \
        --seeds 5 \
        --output-dir results/stress_test
"""

from __future__ import annotations

import abc
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_lyapunov_ppo import LyapunovPPOAgent
from src.env import LatencyEnv

logger = logging.getLogger(__name__)

_LATENCY_BUDGET_MS: float = 50.0
_NUM_FRAMES: int = 2000


# ── Stress scenario base class ──────────────────────────────────────────────

class StressScenario(abc.ABC):
    """Base class for stress-test scenarios.

    A scenario modifies the observation and/or info dict returned by the
    environment at each step to simulate adverse conditions.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable scenario name."""

    @abc.abstractmethod
    def modify_obs(
        self,
        obs: np.ndarray,
        step: int,
        info: Dict[str, Any],
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Apply scenario-specific perturbation to observation and info.

        Parameters
        ----------
        obs : np.ndarray
            Raw (11,) observation from the environment.
        step : int
            Current step index in the episode.
        info : dict
            Step info dict from env.step().

        Returns
        -------
        tuple[np.ndarray, dict]
            Modified (obs, info).
        """

    def reset(self) -> None:
        """Reset any internal state for a new episode."""


# ── Scenario implementations ────────────────────────────────────────────────

class SteadyStateScenario(StressScenario):
    """No perturbation — baseline operating conditions."""

    @property
    def name(self) -> str:
        return "steady_state"

    def modify_obs(self, obs, step, info):
        return obs, info


class DefectBurstScenario(StressScenario):
    """80 % defect rate for 200 frames starting at frame 300."""

    def __init__(self, burst_start: int = 300, burst_len: int = 200,
                 defect_rate: float = 0.80) -> None:
        self._start = burst_start
        self._len = burst_len
        self._rate = defect_rate

    @property
    def name(self) -> str:
        return "defect_burst"

    def modify_obs(self, obs, step, info):
        if self._start <= step < self._start + self._len:
            obs = obs.copy()
            # Increase detection count (index 3), confidence (4), area ratio (5).
            obs[3] = min(1.0, obs[3] + self._rate * 0.6)  # More detections
            obs[4] = min(1.0, max(obs[4], 0.85))           # High confidence
            obs[5] = min(1.0, obs[5] + self._rate * 0.3)  # Larger defect area
            # Latency increases due to processing more detections.
            latency_bump = info.get("latency_ms", 30.0) * 0.25
            info = {**info, "latency_ms": info.get("latency_ms", 30.0) + latency_bump}
        return obs, info


class GPUContentionScenario(StressScenario):
    """Simulated 30 % GPU contention — GPU util spikes, latency increases."""

    def __init__(self, contention_pct: float = 0.30) -> None:
        self._pct = contention_pct

    @property
    def name(self) -> str:
        return "gpu_contention"

    def modify_obs(self, obs, step, info):
        obs = obs.copy()
        # Increase GPU utilization (index 9).
        obs[9] = min(1.0, obs[9] + self._pct)
        # Latency increases proportionally.
        latency_bump = info.get("latency_ms", 30.0) * self._pct
        info = {**info, "latency_ms": info.get("latency_ms", 30.0) + latency_bump}
        return obs, info


class ThermalThrottleScenario(StressScenario):
    """Thermal ramp via sustained load — temperature rises linearly."""

    def __init__(self, ramp_rate: float = 0.0005) -> None:
        self._ramp_rate = ramp_rate

    @property
    def name(self) -> str:
        return "thermal_throttle"

    def modify_obs(self, obs, step, info):
        obs = obs.copy()
        # Temperature ramp (index 10).
        temp_increase = self._ramp_rate * step
        obs[10] = min(1.0, obs[10] + temp_increase)
        # Above 0.8 normalised temp (~86 °C), throttling kicks in.
        if obs[10] > 0.8:
            throttle_factor = (obs[10] - 0.8) * 2.0  # Up to 40 % slowdown
            latency_bump = info.get("latency_ms", 30.0) * throttle_factor
            info = {**info, "latency_ms": info.get("latency_ms", 30.0) + latency_bump}
        return obs, info


class DistributionShiftScenario(StressScenario):
    """Distribution shift via brightness/noise injection at frame 500.

    After the shift point, observations receive Gaussian noise and
    confidence values degrade, simulating out-of-distribution input.
    """

    def __init__(self, shift_frame: int = 500, noise_std: float = 0.15) -> None:
        self._shift = shift_frame
        self._noise_std = noise_std
        self._rng: Optional[np.random.Generator] = None

    @property
    def name(self) -> str:
        return "distribution_shift"

    def reset(self) -> None:
        self._rng = np.random.default_rng(12345)

    def modify_obs(self, obs, step, info):
        if step >= self._shift:
            if self._rng is None:
                self._rng = np.random.default_rng(12345)
            obs = obs.copy()
            noise = self._rng.normal(0, self._noise_std, size=obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0)
            # Degrade confidence.
            obs[4] = max(0.0, obs[4] - 0.2)
            # Latency variability increases.
            lat_noise = abs(self._rng.normal(0, 8.0))
            info = {**info, "latency_ms": info.get("latency_ms", 30.0) + lat_noise}
        return obs, info


class CombinedStressScenario(StressScenario):
    """Combined stress: all stressors overlaid simultaneously."""

    def __init__(self) -> None:
        self._sub = [
            DefectBurstScenario(burst_start=200, burst_len=300),
            GPUContentionScenario(contention_pct=0.20),
            ThermalThrottleScenario(ramp_rate=0.0003),
            DistributionShiftScenario(shift_frame=400, noise_std=0.10),
        ]

    @property
    def name(self) -> str:
        return "combined_stress"

    def reset(self) -> None:
        for s in self._sub:
            s.reset()

    def modify_obs(self, obs, step, info):
        for s in self._sub:
            obs, info = s.modify_obs(obs, step, info)
        return obs, info


# ── All scenarios ────────────────────────────────────────────────────────────

ALL_SCENARIOS: Dict[str, type] = {
    "steady_state": SteadyStateScenario,
    "defect_burst": DefectBurstScenario,
    "gpu_contention": GPUContentionScenario,
    "thermal_throttle": ThermalThrottleScenario,
    "distribution_shift": DistributionShiftScenario,
    "combined_stress": CombinedStressScenario,
}


# ── Per-scenario evaluation ─────────────────────────────────────────────────

def run_scenario(
    agent: LyapunovPPOAgent,
    scenario: StressScenario,
    trace_path: str,
    seed: int,
    num_frames: int,
) -> Dict[str, Any]:
    """Run a single stress scenario and collect metrics."""
    env = LatencyEnv(
        trace_path=trace_path,
        max_steps=num_frames,
        latency_budget_ms=_LATENCY_BUDGET_MS,
    )
    scenario.reset()
    obs, _ = env.reset(seed=seed)

    latencies: List[float] = []
    violations: List[float] = []
    rewards: List[float] = []

    for step in range(num_frames):
        state_tensor = torch.tensor(obs, dtype=torch.float32)
        action, _, _, _ = agent.select_action(state_tensor)

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Apply stress perturbation.
        next_obs, info = scenario.modify_obs(next_obs, step, info)

        latencies.append(info["latency_ms"])
        violations.append(1.0 if info["latency_ms"] > _LATENCY_BUDGET_MS else 0.0)
        rewards.append(reward)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + 1000)

    lat = np.array(latencies)
    viol = np.array(violations)

    return {
        "scenario": scenario.name,
        "seed": seed,
        "num_frames": num_frames,
        "p50_latency_ms": float(np.percentile(lat, 50)),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "mean_latency_ms": float(np.mean(lat)),
        "max_latency_ms": float(np.max(lat)),
        "violation_rate": float(np.mean(viol)),
        "total_violations": int(np.sum(viol)),
        "mean_reward": float(np.mean(rewards)),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stress-test scenarios on Lyapunov-PPO agent.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/ppo_lyapunov"),
        help="Agent checkpoint directory.",
    )
    parser.add_argument(
        "--traces", type=str, default="data/telemetry.parquet",
        help="Path to Parquet trace file.",
    )
    parser.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Scenario names to run (default: all).",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds.")
    parser.add_argument("--frames", type=int, default=2000, help="Frames per scenario run.")
    parser.add_argument("--device", type=str, default="cpu", help="Device string.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/stress_test"),
        help="Output directory.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = parse_args(argv)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build agent.
    config = {
        "ppo": {"gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
                 "entropy_coeff": 0.01, "value_loss_coeff": 0.5,
                 "max_grad_norm": 0.5, "hidden_size": 64},
        "lagrangian": {"lambda_init": 0.1, "lambda_lr": 0.01,
                       "constraint_threshold": 0.01},
        "lyapunov": {"enabled": True, "critic_lr": 3e-4, "drift_tolerance": 0.05},
        "agent": {"checkpoint_dir": str(args.checkpoint)},
    }
    agent = LyapunovPPOAgent(config, device=device)
    agent.load(args.checkpoint)
    logger.info("Loaded agent from %s", args.checkpoint)

    # Select scenarios.
    scenario_names = args.scenarios or list(ALL_SCENARIOS.keys())
    scenarios = [ALL_SCENARIOS[name]() for name in scenario_names]
    logger.info("Running %d scenarios × %d seeds × %d frames",
                len(scenarios), args.seeds, args.frames)

    all_results: List[Dict[str, Any]] = []

    for scenario in scenarios:
        for seed_idx in range(args.seeds):
            seed = seed_idx * 42
            logger.info("  %s — seed %d/%d", scenario.name, seed_idx + 1, args.seeds)
            metrics = run_scenario(agent, scenario, args.traces, seed, args.frames)
            all_results.append(metrics)
            logger.info(
                "    P99=%.2f ms  viol=%.2f%% (%d frames)  reward=%.4f",
                metrics["p99_latency_ms"],
                metrics["violation_rate"] * 100,
                metrics["total_violations"],
                metrics["mean_reward"],
            )

    # Write per-run CSV.
    csv_path = args.output_dir / "stress_per_run.csv"
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    logger.info("Per-run results → %s", csv_path)

    # Aggregate per-scenario.
    agg_rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        name = scenario.name
        runs = [r for r in all_results if r["scenario"] == name]
        row: Dict[str, Any] = {"scenario": name}
        for key in ["p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
                    "mean_latency_ms", "max_latency_ms", "violation_rate",
                    "mean_reward"]:
            vals = [r[key] for r in runs]
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_std"] = float(np.std(vals))
        row["total_violations_mean"] = float(np.mean([r["total_violations"] for r in runs]))
        agg_rows.append(row)

    agg_path = args.output_dir / "stress_summary.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    logger.info("Scenario summary → %s", agg_path)

    # Print summary table.
    logger.info("=" * 80)
    logger.info("STRESS TEST SUMMARY")
    logger.info("-" * 80)
    logger.info("%-22s %8s %8s %8s %8s %10s", "Scenario", "P50", "P95", "P99", "Viol%", "Reward")
    for row in agg_rows:
        logger.info(
            "%-22s %8.2f %8.2f %8.2f %7.2f%% %10.4f",
            row["scenario"],
            row["p50_latency_ms_mean"],
            row["p95_latency_ms_mean"],
            row["p99_latency_ms_mean"],
            row["violation_rate_mean"] * 100,
            row["mean_reward_mean"],
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
