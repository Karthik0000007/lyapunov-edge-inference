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

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d trace records from %d file(s)", len(df), len(parquet_files))

    # Extract state vectors.
    state_cols = [f"state_{i}" for i in range(_STATE_DIM)]
    next_state_cols = [f"next_state_{i}" for i in range(_STATE_DIM)]

    states = df[state_cols].values.astype(np.float32)
    next_states = df[next_state_cols].values.astype(np.float32)
    actions = df["controller_action"].values.astype(np.int64)
    rewards = df["reward"].values.astype(np.float32)
    constraint_costs = df["constraint_cost"].values.astype(np.float32)

    # Rule-based actions for behavioral cloning.
    rule_actions = df["rule_action"].values.astype(np.int64) if "rule_action" in df.columns else actions.copy()

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
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Warm-start the actor via supervised cross-entropy on rule-based actions.

    Returns final action-match accuracy.
    """
    logger.info("Phase 1: Behavioral cloning — %d samples, %d epochs", len(states), epochs)

    s_t = torch.tensor(states, dtype=torch.float32, device=device)
    a_t = torch.tensor(rule_actions, dtype=torch.long, device=device)
    n = len(states)

    actor = agent.actor
    optimizer = optim.Adam(actor.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        actor.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = actor(s_t[idx])
            loss = criterion(logits, a_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Accuracy evaluation.
        actor.eval()
        with torch.no_grad():
            all_logits = actor(s_t)
            preds = all_logits.argmax(dim=1)
            accuracy = (preds == a_t).float().mean().item()

        avg_loss = epoch_loss / max(n_batches, 1)
        best_acc = max(best_acc, accuracy)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  BC Epoch %3d/%d — loss=%.4f  accuracy=%.2f%%",
                epoch, epochs, avg_loss, accuracy * 100,
            )

    logger.info("Phase 1 complete — best accuracy=%.2f%%", best_acc * 100)
    return best_acc


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
) -> Dict[str, List[float]]:
    """PPO + Lagrangian training on trace replay via LatencyEnv.

    Returns dict of training metric histories.
    """
    logger.info("Phase 2: PPO+Lagrangian — %d epochs, rollout=%d", epochs, rollout_length)

    # Create env from trace data.
    traces_dir = checkpoint_dir.parent / "traces_replay"
    traces_dir.mkdir(parents=True, exist_ok=True)
    _write_trace_for_env(traces, traces_dir / "replay.parquet")

    env = LatencyEnv(
        trace_path=str(traces_dir / "replay.parquet"),
        max_steps=rollout_length,
        latency_budget_ms=50.0,
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
            rewards_list, values_list, dones_list, next_value,
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
                idx = perm[start:start + minibatch_size]
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
                epoch, epochs,
                mean_reward,
                violation_rate * 100,
                epoch_metrics.get("lambda", 0.0),
                epoch_metrics.get("policy_loss", 0.0),
                epoch_metrics.get("value_loss", 0.0),
                epoch_metrics.get("entropy", 0.0),
            )

        # Periodic checkpoint.
        if epoch % checkpoint_interval == 0:
            agent.save(checkpoint_dir)
            logger.info("  Checkpoint saved at epoch %d", epoch)

    # Final save.
    agent.save(checkpoint_dir)
    logger.info("Phase 2 complete — final checkpoint saved to %s", checkpoint_dir)

    return metrics


def _write_trace_for_env(traces: Dict[str, np.ndarray], path: Path) -> None:
    """Write trace arrays into a Parquet file compatible with LatencyEnv."""
    import pandas as pd

    n = len(traces["states"])
    resolution_map = {0.0: 320, 0.5: 480, 1.0: 640}

    df = pd.DataFrame({
        "latency_ms": np.random.default_rng(42).normal(30, 10, n).clip(5, 100),
        "detection_count": (traces["states"][:, 3] * 50).astype(int),
        "mean_confidence": traces["states"][:, 4],
        "defect_area_ratio": traces["states"][:, 5],
        "gpu_util_percent": traces["states"][:, 9] * 100,
        "gpu_temp_celsius": traces["states"][:, 10] * 70 + 30,
        "resolution_active": [resolution_map.get(round(v * 2) / 2, 640)
                              for v in traces["states"][:, 6]],
        "segmentation_active": traces["states"][:, 8].astype(bool),
        "threshold_active": traces["states"][:, 7] * 0.2 + 0.25,
    })
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
    parser.add_argument("--bc-epochs", type=int, default=50, help="Behavioral cloning epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--constraint-threshold",
        type=float,
        default=0.01,
        help="Lagrangian constraint threshold d₀.",
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=1000,
        help="Rollout length per PPO epoch.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
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
            "hidden_size": 64,
        },
        "lagrangian": {
            "lambda_init": 0.1,
            "lambda_lr": 0.01,
            "constraint_threshold": args.constraint_threshold,
        },
        "lyapunov": {
            "enabled": False,
            "critic_lr": 3e-4,
            "drift_tolerance": 0.0,
        },
        "agent": {
            "checkpoint_dir": str(args.output_dir),
        },
    }
    agent = LyapunovPPOAgent(config, device=device)

    # Phase 1: Behavioral cloning.
    bc_accuracy = behavioral_cloning(
        agent,
        traces["states"],
        traces["rule_actions"],
        epochs=args.bc_epochs,
        lr=args.lr,
        device=device,
    )
    logger.info("BC accuracy: %.2f%%", bc_accuracy * 100)

    # Phase 2: PPO + Lagrangian.
    metrics = train_ppo(
        agent,
        traces,
        epochs=args.epochs,
        rollout_length=args.rollout_length,
        minibatch_size=args.batch_size,
        device=device,
        checkpoint_dir=args.output_dir,
    )

    # Report final metrics.
    if metrics["violation_rate"]:
        final_viol = metrics["violation_rate"][-1]
        logger.info("Final violation rate: %.2f%%", final_viol * 100)


if __name__ == "__main__":
    main()
