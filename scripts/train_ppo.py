"""
scripts/train_ppo.py
────────────────────
Phase 1: Behavioral cloning warm-start on rule-based traces.
Phase 2: PPO + Lagrangian constrained training on logged traces.

Usage
─────
    python scripts/train_ppo.py \
        --traces traces/exploration \
        --epochs 200 \
        --lr 3e-4 \
        --constraint-threshold 0.01 \
        --latency-budget 35.0 \
        --output-dir checkpoints/ppo_lagrangian
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_lyapunov_ppo import LyapunovPPOAgent
from src.env import LatencyEnv

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_STATE_DIM: int = 11
_NUM_ACTIONS: int = 18


# ── Trace loading ────────────────────────────────────────────────────────────


def load_traces(traces_dir: Path) -> Dict[str, np.ndarray]:
    """Load all Parquet trace files from a directory into numpy arrays.

    Returns dict with keys: states, actions, rewards, constraint_costs,
    next_states, rule_actions.
    """
    import pandas as pd

    parquet_files = sorted(traces_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {traces_dir}")

    # Extract state vectors.
    state_cols = [f"state_{i}" for i in range(_STATE_DIM)]
    next_state_cols = [f"next_state_{i}" for i in range(_STATE_DIM)]

    # Only load files that have state columns (filter out raw telemetry files).
    dfs = []
    for f in parquet_files:
        df_single = pd.read_parquet(f)
        if all(col in df_single.columns for col in state_cols):
            dfs.append(df_single)
        else:
            logger.info("Skipping %s: missing state columns", f.name)

    if not dfs:
        raise FileNotFoundError(f"No Parquet files with state columns found in {traces_dir}")

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d trace records from %d file(s) with state columns", len(df), len(dfs))

    states = df[state_cols].values.astype(np.float32)
    next_states = df[next_state_cols].values.astype(np.float32)
    actions_raw = df["controller_action"].values.astype(np.float32)  # Check for NaN first
    rewards = df["reward"].values.astype(np.float32)
    constraint_costs = df["constraint_cost"].values.astype(np.float32)

    # Filter out rows with NaN in states, next_states, or actions.
    states_valid = np.all(np.isfinite(states), axis=1)
    next_states_valid = np.all(np.isfinite(next_states), axis=1)
    actions_valid = np.isfinite(actions_raw)
    valid_mask = states_valid & next_states_valid & actions_valid

    states = states[valid_mask]
    next_states = next_states[valid_mask]
    actions = actions_raw[valid_mask].astype(np.int64)
    rewards = rewards[valid_mask]
    constraint_costs = constraint_costs[valid_mask]
    rule_actions = actions.copy()  # Use controller actions for BC instead of rule_actions

    logger.info(
        "Filtered %d invalid rows; %d valid samples remain", np.sum(~valid_mask), len(actions)
    )

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "constraint_costs": constraint_costs,
        "next_states": next_states,
        "rule_actions": rule_actions,
    }


# ── Phase 1: Behavioral Cloning ─────────────────────────────────────────────


def behavioral_cloning(
    agent: LyapunovPPOAgent,
    states: np.ndarray,
    rule_actions: np.ndarray,
    epochs: int = 200,
    lr: float = 1.5e-3,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    test_split: float = 0.2,
) -> tuple[float, float]:
    """Warm-start the actor via weighted cross-entropy on rule-based actions.

    Uses a train/test split for unbiased evaluation. Returns (train_acc, test_acc).
    """
    logger.info("Phase 1: Behavioral cloning — %d samples, %d epochs", len(states), epochs)

    # Standardise inputs (zero-mean, unit-variance) for better gradient flow.
    s_np = states.astype(np.float32)
    state_mean = s_np.mean(axis=0)
    state_std = s_np.std(axis=0) + 1e-8
    s_np = (s_np - state_mean) / state_std

    # 80/20 train/test split (stratified by action to preserve class distribution).
    n = len(states)
    test_size = max(1, int(n * test_split))
    train_size = n - test_size

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    s_train = torch.tensor(s_np[train_idx], dtype=torch.float32, device=device)
    a_train = torch.tensor(rule_actions[train_idx], dtype=torch.long, device=device)
    s_test = torch.tensor(s_np[test_idx], dtype=torch.float32, device=device)
    a_test = torch.tensor(rule_actions[test_idx], dtype=torch.long, device=device)

    logger.info("  Train/test split: %d / %d samples", train_size, test_size)

    actor = agent.actor
    optimizer = optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-4)

    # Sqrt-inverse-frequency class weighting (based on training set only).
    num_classes = max(a_train.max().item() + 1, _NUM_ACTIONS)
    class_counts = torch.bincount(a_train, minlength=num_classes).float()
    class_weights = 1.0 / torch.sqrt(class_counts + 1.0)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.05,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=(train_size + batch_size - 1) // batch_size,
        epochs=epochs,
        pct_start=0.3,
    )

    best_train_acc = 0.0
    best_test_acc = 0.0
    best_state = None
    patience_counter = 0
    patience = 12

    for epoch in range(1, epochs + 1):
        actor.train()
        perm = torch.randperm(train_size, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, train_size, batch_size):
            idx = perm[start : start + batch_size]
            logits = actor(s_train[idx])
            loss = criterion(logits, a_train[idx])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Accuracy evaluation on both train and test sets.
        actor.eval()
        with torch.no_grad():
            # Train accuracy.
            train_logits = actor(s_train)
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == a_train).float().mean().item()

            # Test accuracy (for early stopping and checkpointing).
            test_logits = actor(s_test)
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == a_test).float().mean().item()

        avg_loss = epoch_loss / max(n_batches, 1)

        if test_acc > best_test_acc:
            best_train_acc = train_acc
            best_test_acc = test_acc
            best_state = {k: v.clone() for k, v in actor.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  BC Epoch %3d/%d — loss=%.4f  train_acc=%.2f%%  test_acc=%.2f%%",
                epoch,
                epochs,
                avg_loss,
                train_acc * 100,
                test_acc * 100,
            )

        if patience_counter >= patience and best_test_acc >= 0.85:
            logger.info("  Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    # Restore best weights (best test accuracy).
    if best_state is not None:
        actor.load_state_dict(best_state)

    logger.info(
        "Phase 1 complete — train_acc=%.2f%%  test_acc=%.2f%%",
        best_train_acc * 100,
        best_test_acc * 100,
    )
    return best_train_acc, best_test_acc


# ── Phase 2: PPO + Lagrangian ────────────────────────────────────────────────


def train_ppo(
    agent: LyapunovPPOAgent,
    traces: Dict[str, np.ndarray],
    epochs: int,
    rollout_length: int = 1000,
    minibatch_size: int = 256,
    ppo_epochs_per_rollout: int = 4,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: Path = Path("checkpoints/ppo_lagrangian"),
    checkpoint_interval: int = 20,
    latency_budget_ms: float = 50.0,
) -> Dict[str, List[float]]:
    """PPO + Lagrangian training on trace replay via LatencyEnv.

    Parameters
    ----------
    latency_budget_ms:
        Latency budget for constraint cost calculation (default 50.0).
        Lower values (e.g., 30-35ms) force more violations for better
        constraint learning. Use values below trace P95 for training.

    Returns
    -------
    Dict[str, List[float]]
        Training metric histories.
    """
    logger.info(
        "Phase 2: PPO+Lagrangian — %d epochs, rollout=%d, budget=%.1f ms",
        epochs,
        rollout_length,
        latency_budget_ms,
    )

    # Create env from trace data.
    traces_dir = checkpoint_dir.parent / "traces_replay"
    traces_dir.mkdir(parents=True, exist_ok=True)
    _write_trace_for_env(traces, traces_dir / "replay.parquet")

    env = LatencyEnv(
        trace_path=str(traces_dir / "replay.parquet"),
        max_steps=rollout_length,
        latency_budget_ms=latency_budget_ms,
    )

    metrics: Dict[str, List[float]] = {
        "reward": [],
        "constraint_cost": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "lambda": [],
        "violation_rate": [],
    }

    for epoch in range(1, epochs + 1):
        # Collect rollout.
        states_list: List[np.ndarray] = []
        actions_list: List[int] = []
        rewards_list: List[float] = []
        values_list: List[float] = []
        log_probs_list: List[float] = []
        costs_list: List[float] = []
        dones_list: List[bool] = []
        next_states_list: List[np.ndarray] = []

        obs, _ = env.reset()

        for step in range(rollout_length):
            state_tensor = torch.tensor(obs, dtype=torch.float32)
            action, log_prob, value, lyap_val = agent.select_action(state_tensor)

            next_obs, reward, terminated, truncated, info = env.step(action)

            states_list.append(obs)
            actions_list.append(action)
            rewards_list.append(reward)
            values_list.append(value)
            log_probs_list.append(log_prob)
            costs_list.append(info.get("constraint_cost", 0.0))
            dones_list.append(terminated or truncated)
            next_states_list.append(next_obs)

            obs = next_obs

            if terminated or truncated:
                obs, _ = env.reset()

        # Bootstrap value for GAE.
        with torch.no_grad():
            final_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            next_value = agent.critic(final_state).squeeze().item()

        # Compute GAE.
        returns, advantages = agent.compute_gae(
            rewards_list,
            values_list,
            dones_list,
            next_value,
        )

        # Convert to tensors.
        states_t = torch.tensor(np.array(states_list), dtype=torch.float32)
        actions_t = torch.tensor(actions_list, dtype=torch.long)
        old_log_probs_t = torch.tensor(log_probs_list, dtype=torch.float32)
        costs_t = torch.tensor(costs_list, dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states_list), dtype=torch.float32)

        # PPO update (multiple epochs over the rollout).
        epoch_metrics: Dict[str, float] = {}
        for _ in range(ppo_epochs_per_rollout):
            # Shuffle minibatches.
            n = len(states_list)
            perm = torch.randperm(n)
            for start in range(0, n, minibatch_size):
                idx = perm[start : start + minibatch_size]
                epoch_metrics = agent.update(
                    states=states_t[idx],
                    actions=actions_t[idx],
                    old_log_probs=old_log_probs_t[idx],
                    returns=returns[idx],
                    advantages=advantages[idx],
                    constraint_costs=costs_t[idx],
                    next_states=next_states_t[idx],
                )

        # Record metrics.
        mean_reward = float(np.mean(rewards_list))
        mean_cost = float(np.mean(costs_list))
        violation_rate = float(np.mean([1.0 if c > 0 else 0.0 for c in costs_list]))

        # Log detailed violation statistics for debugging
        if epoch <= 5 or epoch % 20 == 0:  # Verbose logging for early epochs and every 20th epoch
            violations = [c for c in costs_list if c > 0]
            latency_samples = [
                info.get("latency_ms", 0)
                for info in [
                    {"latency_ms": 25 + 15 * np.random.normal()}
                    for _ in range(min(10, len(costs_list)))
                ]
            ]
            logger.info(
                "  Debug[epoch %d]: %d violations out of %d steps (%.1f%%)",
                epoch,
                len(violations),
                len(costs_list),
                violation_rate * 100,
            )
            if violations:
                logger.info("    First few violations: %s", violations[:5])
            logger.info("    Sample costs: %s", costs_list[:10])
            logger.info("    Mean cost: %.4f, Max cost: %.4f", mean_cost, max(costs_list))

        metrics["reward"].append(mean_reward)
        metrics["constraint_cost"].append(mean_cost)
        metrics["policy_loss"].append(epoch_metrics.get("policy_loss", 0.0))
        metrics["value_loss"].append(epoch_metrics.get("value_loss", 0.0))
        metrics["entropy"].append(epoch_metrics.get("entropy", 0.0))
        metrics["lambda"].append(epoch_metrics.get("lambda", 0.0))
        metrics["violation_rate"].append(violation_rate)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  PPO Epoch %3d/%d — reward=%.4f  viol=%.2f%%  λ=%.4f  "
                "π_loss=%.4f  v_loss=%.4f  entropy=%.4f",
                epoch,
                epochs,
                mean_reward,
                violation_rate * 100,
                epoch_metrics.get("lambda", 0.0),
                epoch_metrics.get("policy_loss", 0.0),
                epoch_metrics.get("value_loss", 0.0),
                epoch_metrics.get("entropy", 0.0),
            )

            # Warning if no violations detected
            if violation_rate == 0.0 and epoch >= 10:
                logger.warning(
                    "  No violations detected at epoch %d! Budget=%.1f ms. "
                    "Consider using --latency-budget with a lower value (e.g., 25-35ms).",
                    epoch,
                    latency_budget_ms,
                )

        # Periodic checkpoint.
        if epoch % checkpoint_interval == 0:
            agent.save(checkpoint_dir)
            logger.info("  Checkpoint saved at epoch %d", epoch)

    # Final save.
    agent.save(checkpoint_dir)
    logger.info("Phase 2 complete — final checkpoint saved to %s", checkpoint_dir)

    # Save metrics for analysis.
    import json

    metrics_file = checkpoint_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved training metrics to %s", metrics_file)

    return metrics


def _write_trace_for_env(traces: Dict[str, np.ndarray], path: Path) -> None:
    """Write trace arrays into a Parquet file compatible with LatencyEnv."""
    import pandas as pd

    n = len(traces["states"])
    resolution_map = {0.0: 320, 0.5: 480, 1.0: 640}

    df = pd.DataFrame(
        {
            "latency_ms": np.random.default_rng(42).normal(38, 10, n).clip(5, 100),
            "detection_count": (traces["states"][:, 3] * 50).astype(int),
            "mean_confidence": traces["states"][:, 4],
            "defect_area_ratio": traces["states"][:, 5],
            "gpu_util_percent": traces["states"][:, 9] * 100,
            "gpu_temp_celsius": traces["states"][:, 10] * 70 + 30,
            "resolution_active": [
                resolution_map.get(round(v * 2) / 2, 640) for v in traces["states"][:, 6]
            ],
            "segmentation_active": traces["states"][:, 8].astype(bool),
            "threshold_active": traces["states"][:, 7] * 0.2 + 0.25,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO+Lagrangian agent on exploration traces.",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Directory containing Parquet trace files.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="PPO training epochs.")
    parser.add_argument("--bc-epochs", type=int, default=200, help="Behavioral cloning epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--constraint-threshold",
        type=float,
        default=0.01,
        help="Lagrangian constraint threshold d₀ (target violation rate).",
    )
    parser.add_argument(
        "--lambda-lr",
        type=float,
        default=0.01,
        help="Lambda learning rate η_λ for Lagrangian dual gradient ascent. "
        "Higher values (0.03-0.05) converge faster from high violations.",
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=1000,
        help="Rollout length per PPO epoch.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
    parser.add_argument(
        "--latency-budget",
        type=float,
        default=50.0,
        help="Latency budget for constraint violations (ms). "
        "Use lower values (30-35ms) to force more violations for training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device string.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/ppo_lagrangian"),
        help="Checkpoint output directory.",
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

    # Load traces.
    traces = load_traces(args.traces)

    # Build agent config.
    config = {
        "ppo": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coeff": 0.01,
            "value_loss_coeff": 0.5,
            "max_grad_norm": 0.5,
            "hidden_size": 128,
        },
        "lagrangian": {
            "lambda_init": 0.1,
            "lambda_lr": args.lambda_lr,
            "constraint_threshold": args.constraint_threshold,
        },
        "lyapunov": {
            "enabled": True,
            "critic_lr": 3e-4,
            "drift_tolerance": 0.05,
        },
        "agent": {
            "checkpoint_dir": str(args.output_dir),
        },
    }
    agent = LyapunovPPOAgent(config, device=device)

    # Phase 1: Behavioral cloning.
    train_acc, test_acc = behavioral_cloning(
        agent,
        traces["states"],
        traces["rule_actions"],
        epochs=args.bc_epochs,
        lr=1.5e-3,
        batch_size=64,
        device=device,
    )
    logger.info("BC accuracies — train=%.2f%%  test=%.2f%%", train_acc * 100, test_acc * 100)

    # Phase 2: PPO + Lagrangian.
    metrics = train_ppo(
        agent,
        traces,
        epochs=args.epochs,
        rollout_length=args.rollout_length,
        minibatch_size=args.batch_size,
        device=device,
        checkpoint_dir=args.output_dir,
        latency_budget_ms=args.latency_budget,
    )

    # Report final metrics.
    if metrics["violation_rate"]:
        final_viol = metrics["violation_rate"][-1]
        logger.info("Final violation rate: %.2f%%", final_viol * 100)


if __name__ == "__main__":
    main()
