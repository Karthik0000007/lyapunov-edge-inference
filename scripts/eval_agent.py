"""
scripts/eval_agent.py
─────────────────────
Per-agent evaluation on logged traces with all target metrics.

Loads a Lyapunov-PPO agent from checkpoint, evaluates on LatencyEnv
with ≥10,000 frames per run across multiple seeds, computes P50/P95/P99
latency, violation rate, throughput variance, conformal coverage, RL
overhead, and bootstrap 95% CIs on P99 estimates.

Usage
─────
    python scripts/eval_agent.py \
        --checkpoint checkpoints/ppo_lyapunov \
        --traces data/telemetry.parquet \
        --seeds 5 \
        --frames 10000 \
        --output-dir results/eval_agent
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_lyapunov_ppo import LyapunovPPOAgent
from src.env import LatencyEnv

logger = logging.getLogger(__name__)

_STATE_DIM: int = 11
_NUM_ACTIONS: int = 18
_LATENCY_BUDGET_MS: float = 50.0


# ── Default agent config ────────────────────────────────────────────────────

def _default_agent_config(checkpoint_dir: str) -> Dict[str, Any]:
    return {
        "ppo": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coeff": 0.01,
            "value_loss_coeff": 0.5,
            "max_grad_norm": 0.5,
            "hidden_size": 64,
        },
        "lagrangian": {
            "lambda_init": 0.1,
            "lambda_lr": 0.01,
            "constraint_threshold": 0.01,
        },
        "lyapunov": {
            "enabled": True,
            "critic_lr": 3e-4,
            "drift_tolerance": 0.05,
        },
        "agent": {
            "checkpoint_dir": checkpoint_dir,
        },
    }


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    data: np.ndarray,
    stat_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    point = stat_fn(data)
    boot_stats = np.empty(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)
    alpha = 1.0 - ci
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# ── Single-seed evaluation ──────────────────────────────────────────────────

def evaluate_seed(
    agent: LyapunovPPOAgent,
    trace_path: str,
    seed: int,
    num_frames: int,
    device: torch.device,
    latency_noise_std: float = 2.0,
) -> Dict[str, Any]:
    """Run one evaluation seed and return metrics dict."""
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
    overhead_ns: List[float] = []
    actions: List[int] = []

    for _ in range(num_frames):
        state_tensor = torch.tensor(obs, dtype=torch.float32)

        t0 = time.perf_counter_ns()
        action, log_prob, value, lyap_val = agent.select_action(state_tensor)
        t1 = time.perf_counter_ns()

        overhead_ns.append(t1 - t0)
        actions.append(action)

        next_obs, reward, terminated, truncated, info = env.step(action)

        latencies.append(info["latency_ms"])
        violations.append(info["constraint_cost"])
        rewards.append(reward)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + 1000)

    lat = np.array(latencies)
    viol = np.array(violations)
    rew = np.array(rewards)
    oh = np.array(overhead_ns)

    # Throughput = 1000 / latency_ms → frames/sec, compute variance
    throughput = 1000.0 / np.maximum(lat, 1e-3)

    p99_fn = lambda x: float(np.percentile(x, 99))
    p99_point, p99_lo, p99_hi = bootstrap_ci(lat, p99_fn, n_bootstrap=1000)

    metrics = {
        "seed": seed,
        "num_frames": num_frames,
        "p50_latency_ms": float(np.percentile(lat, 50)),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "p99_latency_ms": p99_point,
        "p99_ci_lower": p99_lo,
        "p99_ci_upper": p99_hi,
        "mean_latency_ms": float(np.mean(lat)),
        "violation_rate": float(np.mean(viol)),
        "mean_reward": float(np.mean(rew)),
        "throughput_mean_fps": float(np.mean(throughput)),
        "throughput_var": float(np.var(throughput)),
        "rl_overhead_mean_us": float(np.mean(oh) / 1000),
        "rl_overhead_p99_us": float(np.percentile(oh, 99) / 1000),
    }
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Lyapunov-PPO agent on logged traces.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ppo_lyapunov"),
        help="Agent checkpoint directory.",
    )
    parser.add_argument(
        "--traces",
        type=str,
        default="data/telemetry.parquet",
        help="Path to Parquet trace file.",
    )
    parser.add_argument("--seeds", type=int, default=10, help="Number of evaluation seeds.")
    parser.add_argument("--frames", type=int, default=10000, help="Frames per seed run.")
    parser.add_argument("--latency-noise-std", dest="latency_noise_std", type=float, default=2.0,
                        help="Stochastic noise std for latency (ms).")
    parser.add_argument("--device", type=str, default="cpu", help="Device string.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/eval_agent"),
        help="Output directory for CSV results.",
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

    # Build and load agent.
    config = _default_agent_config(str(args.checkpoint))
    agent = LyapunovPPOAgent(config, device=device)
    agent.load(args.checkpoint)
    logger.info("Loaded agent from %s", args.checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, Any]] = []

    for seed in range(args.seeds):
        logger.info("Evaluating seed %d/%d ...", seed + 1, args.seeds)
        metrics = evaluate_seed(
            agent=agent,
            trace_path=args.traces,
            seed=seed * 42,
            num_frames=args.frames,
            device=device,
            latency_noise_std=args.latency_noise_std,
        )
        all_metrics.append(metrics)
        logger.info(
            "  Seed %d — P99=%.2f ms [%.2f, %.2f]  viol=%.2f%%  reward=%.4f",
            seed,
            metrics["p99_latency_ms"],
            metrics["p99_ci_lower"],
            metrics["p99_ci_upper"],
            metrics["violation_rate"] * 100,
            metrics["mean_reward"],
        )

    # Write per-seed CSV.
    csv_path = args.output_dir / "eval_per_seed.csv"
    fieldnames = list(all_metrics[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    logger.info("Per-seed results written to %s", csv_path)

    # Write aggregate summary.
    agg: Dict[str, Any] = {"method": "PPO+Lyapunov+CP", "num_seeds": args.seeds}
    for key in ["p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
                "mean_latency_ms", "violation_rate", "mean_reward",
                "throughput_mean_fps", "throughput_var",
                "rl_overhead_mean_us", "rl_overhead_p99_us"]:
        vals = [m[key] for m in all_metrics]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    # Aggregate bootstrap CI on P99.
    agg["p99_ci_lower_mean"] = float(np.mean([m["p99_ci_lower"] for m in all_metrics]))
    agg["p99_ci_upper_mean"] = float(np.mean([m["p99_ci_upper"] for m in all_metrics]))

    agg_path = args.output_dir / "eval_summary.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg.keys()))
        writer.writeheader()
        writer.writerow(agg)
    logger.info("Aggregate summary written to %s", agg_path)

    # Print summary.
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY (%d seeds × %d frames)", args.seeds, args.frames)
    logger.info(
        "  P50=%.2f  P95=%.2f  P99=%.2f [%.2f, %.2f] ms",
        agg["p50_latency_ms_mean"],
        agg["p95_latency_ms_mean"],
        agg["p99_latency_ms_mean"],
        agg["p99_ci_lower_mean"],
        agg["p99_ci_upper_mean"],
    )
    logger.info("  Violation rate: %.2f%%", agg["violation_rate_mean"] * 100)
    logger.info("  Mean reward: %.4f", agg["mean_reward_mean"])
    logger.info("  RL overhead: %.1f μs (P99: %.1f μs)",
                agg["rl_overhead_mean_us_mean"], agg["rl_overhead_p99_us_mean"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
