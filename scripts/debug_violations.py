#!/usr/bin/env python3
"""
scripts/debug_violations.py
────────────────────────────
Diagnostic script to test violation tracking and constraint cost calculation.
Forces the environment to produce violations to verify the tracking system works.

Usage:
    python scripts/debug_violations.py --traces traces/ --budget 30.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env import LatencyEnv

logger = logging.getLogger(__name__)


def test_violation_tracking(
    traces_dir: Path,
    budget_ms: float = 30.0,
    n_episodes: int = 5,
    steps_per_episode: int = 200,
) -> None:
    """Test violation tracking with different latency budgets and action strategies."""

    # Find trace files
    trace_files = list(traces_dir.glob("*.parquet"))
    if not trace_files:
        raise FileNotFoundError(f"No trace files found in {traces_dir}")

    trace_file = trace_files[0]  # Use first available trace
    logger.info("Using trace file: %s", trace_file)

    # Check original latency distribution
    df = pd.read_parquet(trace_file)
    if "latency_ms" in df.columns:
        latencies = df["latency_ms"]
        logger.info("Original trace latency distribution:")
        logger.info("  Mean: %.2f ms", latencies.mean())
        logger.info("  P95:  %.2f ms", latencies.quantile(0.95))
        logger.info("  P99:  %.2f ms", latencies.quantile(0.99))
        logger.info("  Max:  %.2f ms", latencies.max())
        logger.info("  >%.1f ms: %.1f%%", budget_ms, (latencies > budget_ms).mean() * 100)

    # Test different action strategies
    strategies = [
        ("conservative", [0, 1, 2]),  # Low-cost actions
        ("aggressive", [15, 16, 17]),  # High-cost actions
        ("mixed", [0, 8, 16]),  # Mix of low/high cost
        ("random", None),  # Random actions
    ]

    results = {}

    for strategy_name, action_list in strategies:
        logger.info("\n" + "=" * 60)
        logger.info("Testing strategy: %s", strategy_name.upper())
        logger.info("=" * 60)

        env = LatencyEnv(
            trace_path=trace_file,
            max_steps=steps_per_episode,
            latency_budget_ms=budget_ms,
        )

        episode_violations = []
        episode_latencies = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            violations = 0
            latencies = []

            for step in range(steps_per_episode):
                if action_list is None:  # Random strategy
                    action = np.random.randint(0, 18)
                else:  # Use predefined actions cyclically
                    action = action_list[step % len(action_list)]

                obs, reward, terminated, truncated, info = env.step(action)

                constraint_cost = info.get("constraint_cost", 0.0)
                latency = info.get("latency_ms", 0.0)

                if constraint_cost > 0:
                    violations += 1
                latencies.append(latency)

                if step < 10:  # Log first 10 steps for debugging
                    logger.debug(
                        "  Step %2d: action=%2d, latency=%.2f ms, violation=%s",
                        step,
                        action,
                        latency,
                        "YES" if constraint_cost > 0 else "no",
                    )

                if terminated or truncated:
                    break

            violation_rate = violations / steps_per_episode
            mean_latency = np.mean(latencies)
            max_latency = np.max(latencies)

            episode_violations.append(violation_rate)
            episode_latencies.append(mean_latency)

            logger.info(
                "  Episode %d: violations=%d/%d (%.1f%%), mean_lat=%.2f ms, max_lat=%.2f ms",
                episode + 1,
                violations,
                steps_per_episode,
                violation_rate * 100,
                mean_latency,
                max_latency,
            )

        # Summary statistics
        mean_viol_rate = np.mean(episode_violations) * 100
        std_viol_rate = np.std(episode_violations) * 100
        mean_latency = np.mean(episode_latencies)

        results[strategy_name] = {
            "violation_rate_pct": mean_viol_rate,
            "violation_std_pct": std_viol_rate,
            "mean_latency_ms": mean_latency,
        }

        logger.info(
            "Strategy '%s' summary: %.1f±%.1f%% violations, %.2f ms avg latency",
            strategy_name,
            mean_viol_rate,
            std_viol_rate,
            mean_latency,
        )

    # Overall summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("Budget: %.1f ms", budget_ms)

    for strategy, metrics in results.items():
        logger.info(
            "%-12s: %5.1f%% violations, %5.2f ms avg latency",
            strategy.capitalize(),
            metrics["violation_rate_pct"],
            metrics["mean_latency_ms"],
        )

    # Check if violation tracking is working
    max_violations = max(results[s]["violation_rate_pct"] for s in results)
    if max_violations == 0.0:
        logger.error("\n❌ NO VIOLATIONS DETECTED WITH ANY STRATEGY!")
        logger.error("This indicates a problem with the violation tracking system.")
        logger.error("Suggestions:")
        logger.error("1. Try even stricter budget (e.g., --budget 20.0)")
        logger.error("2. Check environment's latency interpolation logic")
        logger.error("3. Verify constraint_cost calculation in env.py")
    else:
        logger.info(f"\n✅ Violation tracking is working! Max rate: {max_violations:.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug violation tracking system")
    parser.add_argument(
        "--traces", type=Path, required=True, help="Directory containing trace files"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=30.0,
        help="Latency budget in ms (try lower values like 20-30)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes per strategy")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger.info("Starting violation tracking diagnostic...")
    logger.info("Traces directory: %s", args.traces)
    logger.info("Latency budget: %.1f ms", args.budget)

    test_violation_tracking(
        traces_dir=args.traces,
        budget_ms=args.budget,
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
    )


if __name__ == "__main__":
    main()
