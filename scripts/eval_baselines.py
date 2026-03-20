"""
scripts/eval_baselines.py
─────────────────────────
Multi-baseline comparison framework.

Evaluates all baseline controllers and RL agent variants on LatencyEnv,
computes Wilcoxon signed-rank tests (paired) on per-run P99 between
methods, and produces aggregated comparison tables (CSV + LaTeX).

Usage
─────
    python scripts/eval_baselines.py \
        --traces data/telemetry.parquet \
        --checkpoint checkpoints/ppo_lyapunov \
        --seeds 5 \
        --frames 10000 \
        --output-dir results/eval_baselines
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_lyapunov_ppo import LyapunovPPOAgent
from src.baselines import (
    CONTROLLER_REGISTRY,
    BaseController,
    FixedHighQualityController,
    FixedLowLatencyController,
    PIDController,
    RuleBasedController,
)
from src.env import LatencyEnv
from src.state_features import ControllerState

logger = logging.getLogger(__name__)

_LATENCY_BUDGET_MS: float = 50.0
_STATE_DIM: int = 11


# ── Wrapper to make RL agents behave like BaseController ────────────────────


class _RLControllerWrapper:
    """Wraps a LyapunovPPOAgent to use the same eval loop as baselines."""

    def __init__(self, agent: LyapunovPPOAgent, name: str) -> None:
        self._agent = agent
        self.name = name

    def select_action_index(self, obs: np.ndarray) -> int:
        state_tensor = torch.tensor(obs, dtype=torch.float32)
        action, _, _, _ = self._agent.select_action(state_tensor)
        return action


class _BaselineControllerWrapper:
    """Wraps a BaseController to use observation arrays."""

    def __init__(self, controller: BaseController, name: str) -> None:
        self._ctrl = controller
        self.name = name
        self._latency_window: List[float] = []
        self._window_size: int = 50

    def select_action_index(self, obs: np.ndarray) -> int:
        # Denormalize observation to build ControllerState.
        last_lat = obs[0] * 95.0 + 5.0
        mean_lat = obs[1] * 95.0 + 5.0
        p99_lat = obs[2] * 95.0 + 5.0
        det_count = int(obs[3] * 50.0)
        mean_conf = float(obs[4])
        area_ratio = float(obs[5])
        res_idx = int(round(obs[6] * 2.0))
        thr_idx = int(round(obs[7] * 2.0))
        seg = int(round(obs[8]))
        gpu_util = float(obs[9] * 100.0)
        gpu_temp = float(obs[10] * 70.0 + 30.0)

        state = ControllerState(
            last_latency_ms=last_lat,
            mean_latency_ms=mean_lat,
            p99_latency_ms=p99_lat,
            detection_count=det_count,
            mean_confidence=mean_conf,
            defect_area_ratio=area_ratio,
            resolution_index=res_idx,
            threshold_index=thr_idx,
            segmentation_enabled=seg,
            gpu_utilization=gpu_util,
            gpu_temperature_norm=gpu_temp,
        )
        action = self._ctrl.select_action(state)
        return action.action_index

    def reset(self) -> None:
        self._latency_window.clear()


# ── Single-method evaluation ────────────────────────────────────────────────


def evaluate_method(
    controller,
    trace_path: str,
    seed: int,
    num_frames: int,
    latency_noise_std: float = 2.0,
) -> Dict[str, Any]:
    """Evaluate a single controller for one seed."""
    env = LatencyEnv(
        trace_path=trace_path,
        max_steps=num_frames,
        latency_budget_ms=_LATENCY_BUDGET_MS,
        latency_noise_std=latency_noise_std,
    )
    obs, _ = env.reset(seed=seed)

    latencies: List[float] = []
    violations: List[float] = []
    rewards: List[float] = []

    for _ in range(num_frames):
        action = controller.select_action_index(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        latencies.append(info["latency_ms"])
        violations.append(info["constraint_cost"])
        rewards.append(reward)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + 1000)

    lat = np.array(latencies)
    throughput = 1000.0 / np.maximum(lat, 1e-3)

    return {
        "p50_latency_ms": float(np.percentile(lat, 50)),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "mean_latency_ms": float(np.mean(lat)),
        "violation_rate": float(np.mean(violations)),
        "mean_reward": float(np.mean(rewards)),
        "throughput_mean_fps": float(np.mean(throughput)),
        "throughput_var": float(np.var(throughput)),
    }


# ── Build all methods ───────────────────────────────────────────────────────


def build_methods(
    checkpoint_dir: Path,
    device: torch.device,
) -> List:
    """Build all baseline + RL controllers for comparison."""
    methods = []

    # Non-RL baselines.
    methods.append(_BaselineControllerWrapper(FixedHighQualityController(), "fixed-high"))
    methods.append(_BaselineControllerWrapper(FixedLowLatencyController(), "fixed-low"))
    methods.append(_BaselineControllerWrapper(RuleBasedController(), "rule-based"))
    methods.append(_BaselineControllerWrapper(PIDController(), "pid"))

    # RL variants: PPO-unconstrained (no Lyapunov, no conformal).
    # We load the same checkpoint but the eval loop doesn't use conformal/fallback.
    def _make_rl_agent(name: str, ckpt: Path) -> _RLControllerWrapper:
        config = {
            "ppo": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "entropy_coeff": 0.01,
                "value_loss_coeff": 0.5,
                "max_grad_norm": 0.5,
                "hidden_size": 64,
            },
            "lagrangian": {"lambda_init": 0.1, "lambda_lr": 0.01, "constraint_threshold": 0.01},
            "lyapunov": {"enabled": True, "critic_lr": 3e-4, "drift_tolerance": 0.05},
            "agent": {"checkpoint_dir": str(ckpt)},
        }
        agent = LyapunovPPOAgent(config, device=device)
        agent.load(ckpt)
        return _RLControllerWrapper(agent, name)

    # PPO+Lyapunov (the full agent evaluated raw, without conformal/fallback wrapping).
    methods.append(_make_rl_agent("PPO+Lyapunov", checkpoint_dir))

    return methods


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all baseline controllers and RL agents.",
    )
    parser.add_argument(
        "--traces",
        type=str,
        default="data/telemetry.parquet",
        help="Path to Parquet trace file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ppo_lyapunov"),
        help="RL agent checkpoint directory.",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of evaluation seeds.")
    parser.add_argument("--frames", type=int, default=10000, help="Frames per seed run.")
    parser.add_argument(
        "--latency-noise-std",
        dest="latency_noise_std",
        type=float,
        default=2.0,
        help="Stochastic noise std for latency (ms).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device string.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/eval_baselines"),
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

    methods = build_methods(args.checkpoint, device)
    logger.info(
        "Evaluating %d methods × %d seeds × %d frames", len(methods), args.seeds, args.frames
    )

    # Collect results: method_name → list of per-seed metric dicts.
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for method in methods:
        method_name = method.name
        all_results[method_name] = []
        for seed_idx in range(args.seeds):
            seed = seed_idx * 42
            logger.info("  %s — seed %d/%d", method_name, seed_idx + 1, args.seeds)
            if hasattr(method, "reset"):
                method.reset()
            metrics = evaluate_method(
                method, args.traces, seed, args.frames, args.latency_noise_std
            )
            metrics["method"] = method_name
            metrics["seed"] = seed
            all_results[method_name].append(metrics)
            logger.info(
                "    P99=%.2f ms  viol=%.2f%%  reward=%.4f",
                metrics["p99_latency_ms"],
                metrics["violation_rate"] * 100,
                metrics["mean_reward"],
            )

    # ── Write per-seed CSV ──────────────────────────────────────────────
    all_rows = []
    for method_metrics in all_results.values():
        all_rows.extend(method_metrics)

    csv_path = args.output_dir / "baselines_per_seed.csv"
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("Per-seed results → %s", csv_path)

    # ── Aggregate comparison table ──────────────────────────────────────
    method_names = list(all_results.keys())
    metric_keys = [
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "mean_latency_ms",
        "violation_rate",
        "mean_reward",
        "throughput_mean_fps",
    ]

    agg_rows: List[Dict[str, Any]] = []
    for name in method_names:
        row: Dict[str, Any] = {"method": name}
        seed_metrics = all_results[name]
        for key in metric_keys:
            vals = [m[key] for m in seed_metrics]
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_std"] = float(np.std(vals))
        agg_rows.append(row)

    agg_path = args.output_dir / "baselines_comparison.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    logger.info("Comparison table → %s", agg_path)

    # ── Wilcoxon signed-rank tests (paired on per-run P99) ──────────────
    # Compare every method against the last (PPO+Lyapunov) as reference.
    reference = method_names[-1]
    ref_p99 = np.array([m["p99_latency_ms"] for m in all_results[reference]])

    wilcoxon_rows: List[Dict[str, Any]] = []
    for name in method_names:
        if name == reference:
            continue
        other_p99 = np.array([m["p99_latency_ms"] for m in all_results[name]])
        diff = ref_p99 - other_p99
        if np.all(diff == 0):
            p_val = 1.0
            stat = 0.0
        elif len(diff) < 6:
            # Wilcoxon needs ≥6 non-zero differences for reliability;
            # fall back to sign test approximation.
            n_pos = int(np.sum(diff > 0))
            n_nonzero = int(np.sum(diff != 0))
            if n_nonzero == 0:
                p_val = 1.0
            else:
                p_val = float(stats.binomtest(n_pos, n_nonzero, 0.5).pvalue)
            stat = float(n_pos)
        else:
            stat, p_val = stats.wilcoxon(ref_p99, other_p99)
            stat = float(stat)
            p_val = float(p_val)

        wilcoxon_rows.append(
            {
                "method_A": reference,
                "method_B": name,
                "p99_A_mean": float(np.mean(ref_p99)),
                "p99_B_mean": float(np.mean(other_p99)),
                "wilcoxon_stat": stat,
                "p_value": p_val,
                "significant_005": p_val < 0.05,
            }
        )

    wilcoxon_path = args.output_dir / "wilcoxon_tests.csv"
    if wilcoxon_rows:
        with open(wilcoxon_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(wilcoxon_rows[0].keys()))
            writer.writeheader()
            writer.writerows(wilcoxon_rows)
        logger.info("Wilcoxon tests → %s", wilcoxon_path)

    # ── LaTeX table ─────────────────────────────────────────────────────
    latex_path = args.output_dir / "baselines_comparison.tex"
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline comparison (mean $\\pm$ std across %d seeds)}\n" % args.seeds)
        f.write("\\label{tab:baselines}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & P50 (ms) & P95 (ms) & P99 (ms) & Viol. (\\%) & " "Reward & FPS \\\\\n")
        f.write("\\midrule\n")
        for row in agg_rows:
            name = row["method"].replace("_", "\\_")
            f.write(
                "%s & %.1f$\\pm$%.1f & %.1f$\\pm$%.1f & %.1f$\\pm$%.1f "
                "& %.1f$\\pm$%.1f & %.3f$\\pm$%.3f & %.0f$\\pm$%.0f \\\\\n"
                % (
                    name,
                    row["p50_latency_ms_mean"],
                    row["p50_latency_ms_std"],
                    row["p95_latency_ms_mean"],
                    row["p95_latency_ms_std"],
                    row["p99_latency_ms_mean"],
                    row["p99_latency_ms_std"],
                    row["violation_rate_mean"] * 100,
                    row["violation_rate_std"] * 100,
                    row["mean_reward_mean"],
                    row["mean_reward_std"],
                    row["throughput_mean_fps_mean"],
                    row["throughput_mean_fps_std"],
                )
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    logger.info("LaTeX table → %s", latex_path)

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("BASELINE COMPARISON SUMMARY")
    logger.info("-" * 70)
    logger.info("%-20s %8s %8s %8s %8s %8s", "Method", "P50", "P95", "P99", "Viol%", "Reward")
    for row in agg_rows:
        logger.info(
            "%-20s %8.2f %8.2f %8.2f %7.2f%% %8.4f",
            row["method"],
            row["p50_latency_ms_mean"],
            row["p95_latency_ms_mean"],
            row["p99_latency_ms_mean"],
            row["violation_rate_mean"] * 100,
            row["mean_reward_mean"],
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
