#!/usr/bin/env python3
"""
scripts/explore_config.py
─────────────────────────
Generate training configurations optimized for exploration and violation discovery.

Creates modified config files and training commands that encourage the agent
to explore more aggressively and discover constraint violations.

Usage:
    python scripts/explore_config.py --base-traces traces/ --output-dir config/exploration/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_trace_latencies(traces_dir: Path) -> Dict[str, float]:
    """Analyze latency distribution in trace files to suggest budgets."""
    trace_files = list(traces_dir.glob("*.parquet"))
    if not trace_files:
        logger.warning("No trace files found in %s", traces_dir)
        return {"p50": 30.0, "p95": 45.0, "p99": 50.0, "max": 60.0}

    all_latencies = []
    for trace_file in trace_files:
        try:
            df = pd.read_parquet(trace_file)
            if "latency_ms" in df.columns:
                all_latencies.extend(df["latency_ms"].tolist())
            else:
                logger.warning("No latency_ms column in %s", trace_file)
        except Exception as e:
            logger.warning("Error reading %s: %s", trace_file, e)

    if not all_latencies:
        logger.warning("No latency data found; using default values")
        return {"p50": 30.0, "p95": 45.0, "p99": 50.0, "max": 60.0}

    latencies = np.array(all_latencies)
    return {
        "mean": float(np.mean(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "max": float(np.max(latencies)),
    }


def generate_exploration_configs(latency_stats: Dict[str, float]) -> List[Dict[str, Any]]:
    """Generate multiple exploration-optimized training configurations."""

    # Base configuration
    base_config = {
        "epochs": 200,
        "bc_epochs": 50,  # Reduced BC to allow more exploration
        "lr": 3e-4,
        "rollout_length": 1000,
        "batch_size": 256,
        "constraint_threshold": 0.01,
    }

    configs = []

    # 1. Conservative exploration (slight increase in entropy)
    conservative = base_config.copy()
    conservative.update(
        {
            "name": "conservative_exploration",
            "description": "Slightly increased exploration with moderate budget",
            "latency_budget": max(25.0, latency_stats["p95"] * 0.8),  # 80% of P95
            "entropy_coeff": 0.02,  # Double default entropy
            "clip_epsilon": 0.25,  # Slightly larger policy updates
        }
    )
    configs.append(conservative)

    # 2. Aggressive exploration (high entropy, strict budget)
    aggressive = base_config.copy()
    aggressive.update(
        {
            "name": "aggressive_exploration",
            "description": "High exploration with strict latency budget",
            "latency_budget": max(20.0, latency_stats["p50"] * 1.1),  # Just above median
            "entropy_coeff": 0.05,  # 5x default entropy
            "clip_epsilon": 0.3,  # Larger policy updates
            "epochs": 150,  # Fewer epochs due to aggressive exploration
        }
    )
    configs.append(aggressive)

    # 3. Progressive budget (starts strict, relaxes over time)
    progressive = base_config.copy()
    progressive.update(
        {
            "name": "progressive_budget",
            "description": "Budget starts strict and relaxes during training",
            "latency_budget": max(22.0, latency_stats["p50"] * 1.2),
            "entropy_coeff": 0.03,
            "schedule": {
                "budget_start": max(20.0, latency_stats["p50"] * 0.9),
                "budget_end": max(35.0, latency_stats["p95"] * 0.9),
                "schedule_epochs": 100,  # Relax budget over first 100 epochs
            },
        }
    )
    configs.append(progressive)

    # 4. Curiosity-driven (encourage state visitation diversity)
    curiosity = base_config.copy()
    curiosity.update(
        {
            "name": "curiosity_driven",
            "description": "Encourage diverse state visitation",
            "latency_budget": max(28.0, latency_stats["p95"] * 0.85),
            "entropy_coeff": 0.04,
            "rollout_length": 1500,  # Longer episodes for exploration
            "exploration_bonus": True,  # Add exploration bonus if implemented
        }
    )
    configs.append(curiosity)

    return configs


def create_training_commands(
    configs: List[Dict[str, Any]], traces_dir: Path, output_dir: Path
) -> List[str]:
    """Generate shell commands to run each exploration configuration."""
    commands = []

    for config in configs:
        name = config["name"]
        output_path = output_dir / f"checkpoints_{name}"

        cmd = f"""python scripts/train_ppo.py \\
    --traces {traces_dir} \\
    --epochs {config['epochs']} \\
    --bc-epochs {config['bc_epochs']} \\
    --lr {config['lr']} \\
    --latency-budget {config['latency_budget']:.1f} \\
    --constraint-threshold {config['constraint_threshold']} \\
    --rollout-length {config['rollout_length']} \\
    --batch-size {config['batch_size']} \\
    --output-dir {output_path}"""

        commands.append(cmd)

    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exploration-optimized training configurations"
    )
    parser.add_argument(
        "--base-traces",
        type=Path,
        required=True,
        help="Directory containing trace files for analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("config/exploration"),
        help="Output directory for configs and scripts",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze trace latencies
    logger.info("Analyzing latency distribution in %s...", args.base_traces)
    latency_stats = analyze_trace_latencies(args.base_traces)
    logger.info("Latency analysis: %s", {k: f"{v:.1f}ms" for k, v in latency_stats.items()})

    # Generate exploration configs
    configs = generate_exploration_configs(latency_stats)
    logger.info("Generated %d exploration configurations", len(configs))

    # Save configurations
    for config in configs:
        config_file = args.output_dir / f"{config['name']}.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Saved config: %s", config_file)
        logger.info("  %s", config["description"])
        logger.info(
            "  Budget: %.1f ms, Entropy: %.3f",
            config["latency_budget"],
            config.get("entropy_coeff", 0.01),
        )

    # Generate training scripts
    commands = create_training_commands(configs, args.base_traces, args.output_dir)

    # Write master training script
    script_file = args.output_dir / "run_exploration_experiments.sh"
    with open(script_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Exploration-optimized training experiments\n")
        f.write(f"# Generated from traces: {args.base_traces.resolve()}\n\n")

        for i, (config, cmd) in enumerate(zip(configs, commands), 1):
            f.write(f"# Experiment {i}: {config['name']}\n")
            f.write(f"# {config['description']}\n")
            f.write(f"echo 'Starting experiment: {config['name']}'\n")
            f.write(f"{cmd}\n")
            f.write(f"echo 'Completed: {config['name']}'\n\n")

    # Make script executable (Unix-style, won't affect Windows but doesn't hurt)
    script_file.chmod(0o755)

    logger.info("Created training script: %s", script_file)
    logger.info("\nTo run all exploration experiments:")
    logger.info("  bash %s", script_file)
    logger.info("\nOr run individual experiments:")
    for i, config in enumerate(configs, 1):
        logger.info("  # %d. %s", i, config["description"])

    # Summary
    print("\n" + "=" * 60)
    print("EXPLORATION CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Trace latency P95: {latency_stats['p95']:.1f} ms")
    print("Recommended budget for violations:")
    for config in configs:
        budget = config["latency_budget"]
        entropy = config.get("entropy_coeff", 0.01)
        violation_est = max(
            0, (budget - latency_stats["p50"]) / (latency_stats["p95"] - latency_stats["p50"]) * 30
        )
        print(
            f"  {config['name']:20s}: {budget:5.1f} ms (entropy={entropy:.3f}, ~{violation_est:.1f}% violations)"
        )


if __name__ == "__main__":
    main()
