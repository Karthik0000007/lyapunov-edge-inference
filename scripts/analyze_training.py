#!/usr/bin/env python3
"""
scripts/analyze_training.py
───────────────────────────
Analyze PPO training dynamics to diagnose convergence issues.

Usage:
    python scripts/analyze_training.py --checkpoint checkpoints/ppo_lagrangian
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def load_checkpoint_metrics(checkpoint_dir: Path) -> Dict[str, List[float]]:
    """Load training metrics from checkpoint directory."""
    metrics_file = checkpoint_dir / "metrics.json"

    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)

    # If no metrics file, try to reconstruct from logs
    logger.warning("No metrics.json found, metrics tracking not available")
    return {}


def analyze_convergence(metrics: Dict[str, List[float]]) -> Dict[str, any]:
    """Analyze training convergence and provide diagnostic insights."""
    if not metrics:
        return {"status": "no_data"}

    violation_rates = metrics.get("violation_rate", [])
    lambdas = metrics.get("lambda", [])
    rewards = metrics.get("reward", [])

    if not violation_rates:
        return {"status": "no_violation_data"}

    n_epochs = len(violation_rates)
    final_viol = violation_rates[-1] * 100
    initial_viol = violation_rates[0] * 100

    # Compute trends
    recent_viols = violation_rates[-20:] if len(violation_rates) >= 20 else violation_rates
    mean_recent_viol = np.mean(recent_viols) * 100
    std_recent_viol = np.std(recent_viols) * 100

    # Check if converging
    if len(violation_rates) >= 50:
        first_half = np.mean(violation_rates[:n_epochs//2]) * 100
        second_half = np.mean(violation_rates[n_epochs//2:]) * 100
        improvement_rate = (first_half - second_half) / first_half * 100
    else:
        improvement_rate = 0.0

    # Lambda analysis
    final_lambda = lambdas[-1] if lambdas else 0.0
    initial_lambda = lambdas[0] if lambdas else 0.0
    lambda_growth = final_lambda - initial_lambda

    analysis = {
        "status": "analyzed",
        "epochs": n_epochs,
        "initial_violation_pct": initial_viol,
        "final_violation_pct": final_viol,
        "recent_violation_pct": mean_recent_viol,
        "recent_violation_std": std_recent_viol,
        "improvement_rate_pct": improvement_rate,
        "final_lambda": final_lambda,
        "initial_lambda": initial_lambda,
        "lambda_growth": lambda_growth,
        "target_violation_pct": 1.0,
    }

    # Diagnosis
    if final_viol <= 2.0:
        analysis["diagnosis"] = "converged"
        analysis["recommendation"] = "Training successful! Violation rate within target."
    elif std_recent_viol < 2.0 and improvement_rate < 5.0:
        analysis["diagnosis"] = "plateau"
        analysis["recommendation"] = "Training plateaued. Need higher λ learning rate or more epochs."
    elif improvement_rate > 10.0:
        analysis["diagnosis"] = "improving"
        analysis["recommendation"] = "Training improving. Continue for more epochs."
    else:
        analysis["diagnosis"] = "uncertain"
        analysis["recommendation"] = "Training dynamics unclear. Check hyperparameters."

    return analysis


def plot_training_curves(metrics: Dict[str, List[float]], output_dir: Path) -> None:
    """Generate training curve plots."""
    if not metrics:
        logger.warning("No metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Violation rate
    if "violation_rate" in metrics:
        viols = np.array(metrics["violation_rate"]) * 100
        axes[0, 0].plot(viols, 'b-', alpha=0.7, label='Violation Rate')
        axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Target (1%)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Violation Rate (%)')
        axes[0, 0].set_title('Constraint Violation Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Lambda (Lagrangian multiplier)
    if "lambda" in metrics:
        lambdas = metrics["lambda"]
        axes[0, 1].plot(lambdas, 'g-', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('λ (Lagrangian Multiplier)')
        axes[0, 1].set_title('Constraint Penalty λ')
        axes[0, 1].grid(True, alpha=0.3)

    # Reward
    if "reward" in metrics:
        rewards = metrics["reward"]
        axes[1, 0].plot(rewards, 'purple', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Mean Reward')
        axes[1, 0].set_title('Training Reward')
        axes[1, 0].grid(True, alpha=0.3)

    # Constraint cost
    if "constraint_cost" in metrics:
        costs = metrics["constraint_cost"]
        axes[1, 1].plot(costs, 'orange', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Mean Constraint Cost')
        axes[1, 1].set_title('Constraint Cost')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "training_curves.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info("Saved training curves to: %s", output_file)
    plt.close()


def print_recommendations(analysis: Dict[str, any]) -> None:
    """Print actionable recommendations based on analysis."""
    print("\n" + "="*60)
    print("TRAINING ANALYSIS SUMMARY")
    print("="*60)

    if analysis["status"] != "analyzed":
        print("❌ Unable to analyze: insufficient data")
        return

    print(f"Training epochs:        {analysis['epochs']}")
    print(f"Initial violations:     {analysis['initial_violation_pct']:.1f}%")
    print(f"Final violations:       {analysis['final_violation_pct']:.1f}%")
    print(f"Recent violations:      {analysis['recent_violation_pct']:.1f}% ± {analysis['recent_violation_std']:.1f}%")
    print(f"Target violations:      {analysis['target_violation_pct']:.1f}%")
    print(f"Improvement rate:       {analysis['improvement_rate_pct']:.1f}%")
    print(f"Lambda (λ):             {analysis['initial_lambda']:.3f} → {analysis['final_lambda']:.3f}")

    print(f"\n📊 Diagnosis: {analysis['diagnosis'].upper()}")
    print(f"💡 Recommendation: {analysis['recommendation']}")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    final_viol = analysis['final_violation_pct']

    if final_viol <= 2.0:
        print("✅ SUCCESS! Your constraint learning has converged.")
        print("\nNext steps:")
        print("1. Evaluate agent: python scripts/eval_agent.py --checkpoint checkpoints/ppo_lagrangian")
        print("2. Run stress tests: python scripts/stress_test.py")
        print("3. Generate ablation studies: python scripts/ablation.py")

    elif analysis['diagnosis'] == 'plateau':
        print("🔧 PLATEAU DETECTED - Need stronger constraint pressure")
        print("\nOption 1: Increase λ learning rate (recommended)")
        print("   # Edit src/agent_lyapunov_ppo.py, increase lambda_lr from 0.01 to 0.03")
        print("   python scripts/train_ppo.py --traces traces/ --latency-budget 30.0 --epochs 300")
        print("\nOption 2: Train longer with current settings")
        print("   python scripts/train_ppo.py --traces traces/ --latency-budget 30.0 --epochs 400")
        print("\nOption 3: Stricter constraint threshold")
        print("   python scripts/train_ppo.py --traces traces/ --constraint-threshold 0.005 --epochs 300")

    elif analysis['diagnosis'] == 'improving':
        print("📈 MAKING PROGRESS - Continue training")
        print("\nRecommended:")
        print("   python scripts/train_ppo.py --traces traces/ --latency-budget 30.0 --epochs 400")
        print("\nProjected epochs to target: ~%.0f epochs" %
              (analysis['epochs'] * analysis['final_violation_pct'] / analysis['target_violation_pct']))

    else:
        print("🤔 TRAINING DYNAMICS UNCLEAR")
        print("\nDebug steps:")
        print("1. Check if λ is increasing (should grow over time)")
        print("2. Verify violation rate is decreasing (even slowly)")
        print("3. Try increasing λ learning rate or training longer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PPO training dynamics")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory with metrics"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate training curve plots"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger.info("Analyzing training from: %s", args.checkpoint)

    # Load metrics
    metrics = load_checkpoint_metrics(args.checkpoint)

    if not metrics:
        logger.error("No metrics found in checkpoint directory!")
        logger.error("Make sure training completed and metrics were saved.")
        return

    # Analyze
    analysis = analyze_convergence(metrics)

    # Print recommendations
    print_recommendations(analysis)

    # Plot if requested
    if args.plot:
        plot_training_curves(metrics, args.checkpoint)


if __name__ == "__main__":
    main()
