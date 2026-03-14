"""
scripts/calibrate_cp.py
───────────────────────
Calibrate the conformal prediction safety certificate:
  1. Train the latency predictor MLP via MSE (50 epochs)
  2. Compute nonconformity scores (residuals e_i = ℓ_i − ℓ̂(s_i, a_i))
  3. Extract calibration quantile at 1−α
  4. Verify empirical coverage ≥ 1−α on a held-out validation split
  5. Save predictor weights, quantile, and α to checkpoint
  6. Generate calibration diagnostic plots

Usage
─────
    python scripts/calibrate_cp.py \
        --traces traces/exploration \
        --alpha 0.01 \
        --calibration-size 5000 \
        --output-dir checkpoints/conformal
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal import ConformalPredictor
from src.latency_predictor import LatencyPredictor

logger = logging.getLogger(__name__)

_STATE_DIM: int = 11
_NUM_ACTIONS: int = 18


# ── Trace loading ────────────────────────────────────────────────────────────

def load_calibration_data(
    traces_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load trace data needed for calibration."""
    import pandas as pd

    parquet_files = sorted(traces_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {traces_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d records from %d file(s)", len(df), len(parquet_files))

    state_cols = [f"state_{i}" for i in range(_STATE_DIM)]

    return {
        "states": df[state_cols].values.astype(np.float32),
        "actions": df["controller_action"].values.astype(np.int64),
        "latencies": df["latency_ms"].values.astype(np.float32),
    }


# ── Calibration pipeline ────────────────────────────────────────────────────

def calibrate(
    traces_dir: Path,
    alpha: float,
    calibration_size: int,
    predictor_epochs: int,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Run the full calibration pipeline.

    Returns dict with calibration metrics.
    """
    data = load_calibration_data(traces_dir)
    states = data["states"]
    actions = data["actions"]
    latencies = data["latencies"]

    n = len(states)
    logger.info("Total samples: %d  requested calibration: %d", n, calibration_size)

    # Limit to calibration_size if data is larger.
    if n > calibration_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=calibration_size, replace=False)
        states = states[idx]
        actions = actions[idx]
        latencies = latencies[idx]
        n = calibration_size

    # Train/calibration/validation split: 60% train, 20% calibrate, 20% validate.
    n_train = int(n * 0.6)
    n_cal = int(n * 0.2)
    n_val = n - n_train - n_cal

    perm = np.random.default_rng(42).permutation(n)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train + n_cal]
    val_idx = perm[n_train + n_cal:]

    logger.info("Split: train=%d  calibrate=%d  validate=%d", n_train, n_cal, n_val)

    # ── Step 1: Train latency predictor MLP ──────────────────────────────
    logger.info("Step 1: Training latency predictor MLP (%d epochs)", predictor_epochs)

    predictor_config = {
        "predictor_hidden_size": 32,
        "predictor_checkpoint": str(output_dir / "latency_predictor.pt"),
    }
    predictor = LatencyPredictor(predictor_config, device=device)

    train_losses = predictor.train(
        states=states[train_idx],
        actions=actions[train_idx],
        latencies=latencies[train_idx],
        epochs=predictor_epochs,
        lr=1e-3,
        batch_size=256,
    )

    # Save predictor weights.
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(output_dir / "latency_predictor.pt")
    logger.info("Latency predictor saved to %s", output_dir / "latency_predictor.pt")

    # ── Step 2: Compute nonconformity scores on calibration set ──────────
    logger.info("Step 2: Computing nonconformity scores on calibration set")

    s_cal = torch.tensor(states[cal_idx], dtype=torch.float32, device=device)
    a_cal = torch.tensor(actions[cal_idx], dtype=torch.long, device=device)
    predicted_cal = predictor.predict_batch(s_cal, a_cal).cpu().numpy()

    # Nonconformity scores: e_i = ℓ_i − ℓ̂(s_i, a_i)
    scores = latencies[cal_idx] - predicted_cal

    logger.info(
        "Nonconformity scores — mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        scores.mean(), scores.std(), scores.min(), scores.max(),
    )

    # ── Step 3: Extract calibration quantile at 1−α ──────────────────────
    quantile = float(np.quantile(scores, 1.0 - alpha))
    logger.info(
        "Step 3: Calibration quantile q̂_{1-α} = %.4f ms  (α=%.4f)",
        quantile, alpha,
    )

    # ── Step 4: Verify coverage on validation set ────────────────────────
    logger.info("Step 4: Verifying coverage on validation set (%d samples)", n_val)

    s_val = torch.tensor(states[val_idx], dtype=torch.float32, device=device)
    a_val = torch.tensor(actions[val_idx], dtype=torch.long, device=device)
    predicted_val = predictor.predict_batch(s_val, a_val).cpu().numpy()

    upper_bounds = predicted_val + quantile
    covered = latencies[val_idx] <= upper_bounds
    coverage = float(covered.mean())
    target_coverage = 1.0 - alpha

    logger.info(
        "Empirical coverage: %.4f  (target ≥ %.4f)  %s",
        coverage,
        target_coverage,
        "PASS" if coverage >= target_coverage else "FAIL",
    )

    # Violation analysis.
    violations = ~covered
    if violations.any():
        violation_residuals = latencies[val_idx][violations] - predicted_val[violations]
        logger.info(
            "  Violations: %d/%d  mean excess=%.4f ms  max excess=%.4f ms",
            violations.sum(), n_val,
            violation_residuals.mean() - quantile,
            violation_residuals.max() - quantile,
        )

    # ── Step 5: Save conformal state ─────────────────────────────────────
    conformal_state = {
        "alpha": alpha,
        "quantile": quantile,
        "calibrated": True,
        "scores": scores.tolist(),
    }
    state_path = output_dir / "conformal_state.pt"
    torch.save(conformal_state, state_path)
    logger.info("Conformal state saved to %s", state_path)

    # ── Step 6: Generate calibration diagnostic plots ────────────────────
    _generate_diagnostics(
        scores, predicted_val, latencies[val_idx], upper_bounds,
        coverage, alpha, output_dir,
    )

    return {
        "coverage": coverage,
        "target_coverage": target_coverage,
        "quantile": quantile,
        "alpha": alpha,
        "n_calibration": n_cal,
        "n_validation": n_val,
        "final_train_mse": train_losses[-1] if train_losses else 0.0,
    }


def _generate_diagnostics(
    scores: np.ndarray,
    predicted: np.ndarray,
    actual: np.ndarray,
    bounds: np.ndarray,
    coverage: float,
    alpha: float,
    output_dir: Path,
) -> None:
    """Generate calibration diagnostic plots (saved as PNG)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping diagnostic plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Conformal Prediction Calibration Diagnostics (α={alpha})", fontsize=14)

    # Plot 1: Residual distribution.
    ax = axes[0, 0]
    ax.hist(scores, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    quantile = float(np.quantile(scores, 1.0 - alpha))
    ax.axvline(quantile, color="red", linestyle="--", linewidth=2,
               label=f"q̂(1−α) = {quantile:.2f}")
    ax.set_xlabel("Nonconformity Score (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution (Calibration Set)")
    ax.legend()

    # Plot 2: Predicted vs actual latency.
    ax = axes[0, 1]
    ax.scatter(actual, predicted, alpha=0.3, s=4, color="steelblue")
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y=x")
    ax.set_xlabel("Actual Latency (ms)")
    ax.set_ylabel("Predicted Latency (ms)")
    ax.set_title("Prediction Accuracy (Validation Set)")
    ax.legend()

    # Plot 3: Coverage over sorted samples.
    ax = axes[1, 0]
    sorted_idx = np.argsort(actual)
    cumulative_coverage = np.cumsum(actual[sorted_idx] <= bounds[sorted_idx]) / np.arange(1, len(actual) + 1)
    ax.plot(cumulative_coverage, color="steelblue", linewidth=1)
    ax.axhline(1.0 - alpha, color="red", linestyle="--", linewidth=1,
               label=f"Target = {1.0 - alpha:.2f}")
    ax.axhline(coverage, color="green", linestyle=":", linewidth=1,
               label=f"Achieved = {coverage:.4f}")
    ax.set_xlabel("Sample Index (sorted by actual latency)")
    ax.set_ylabel("Cumulative Coverage")
    ax.set_title("Coverage Verification")
    ax.legend()

    # Plot 4: Upper bound vs actual.
    ax = axes[1, 1]
    n_show = min(200, len(actual))
    x = np.arange(n_show)
    ax.scatter(x, actual[:n_show], s=8, alpha=0.7, color="steelblue", label="Actual")
    ax.scatter(x, bounds[:n_show], s=8, alpha=0.5, color="tomato", label="Upper Bound")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Conformal Bounds (first {n_show} validation samples)")
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / "calibration_diagnostics.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Diagnostic plots saved to %s", plot_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate conformal prediction safety certificate.",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Directory containing Parquet trace files.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Target miscoverage rate α (default 0.01 for 99%% coverage).",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=10000,
        help="Maximum number of samples for calibration.",
    )
    parser.add_argument(
        "--predictor-epochs",
        type=int,
        default=50,
        help="Latency predictor training epochs.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device string.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/conformal"),
        help="Output directory for conformal state and predictor weights.",
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

    results = calibrate(
        traces_dir=args.traces,
        alpha=args.alpha,
        calibration_size=args.calibration_size,
        predictor_epochs=args.predictor_epochs,
        output_dir=args.output_dir,
        device=device,
    )

    logger.info("Calibration complete:")
    logger.info("  Coverage: %.4f  (target ≥ %.4f)", results["coverage"], results["target_coverage"])
    logger.info("  Quantile: %.4f ms", results["quantile"])
    logger.info("  Predictor final MSE: %.6f", results["final_train_mse"])


if __name__ == "__main__":
    main()
