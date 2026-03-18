"""
scripts/train_lyapunov.py
─────────────────────────
Upgrade a pre-trained PPO+Lagrangian agent with:
  - Lyapunov critic L_φ training via TD on logged constraint costs
  - Transition model training on (s, a, s') tuples
  - PPO update with Lyapunov action masking active

Usage
─────
    python scripts/train_lyapunov.py \
        --traces traces/exploration \
        --pretrained checkpoints/ppo_lagrangian \
        --epochs 100 \
        --output-dir checkpoints/ppo_lyapunov
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

from src.agent_lyapunov_ppo import LyapunovPPOAgent
from src.env import LatencyEnv

logger = logging.getLogger(__name__)

_STATE_DIM: int = 11
_NUM_ACTIONS: int = 18


# ── Trace loading (reuse from train_ppo) ────────────────────────────────────

def load_traces(traces_dir: Path) -> Dict[str, np.ndarray]:
    """Load Parquet traces into numpy arrays."""
    import pandas as pd

    parquet_files = sorted(traces_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {traces_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d trace records from %d file(s)", len(df), len(parquet_files))

    state_cols = [f"state_{i}" for i in range(_STATE_DIM)]
    next_state_cols = [f"next_state_{i}" for i in range(_STATE_DIM)]

    return {
        "states": df[state_cols].values.astype(np.float32),
        "actions": df["controller_action"].values.astype(np.int64),
        "rewards": df["reward"].values.astype(np.float32),
        "constraint_costs": df["constraint_cost"].values.astype(np.float32),
        "next_states": df[next_state_cols].values.astype(np.float32),
    }


# ── Phase 3a: Transition model pre-training ─────────────────────────────────

def train_transition_model(
    agent: LyapunovPPOAgent,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> List[float]:
    """Train the transition model T(s, a) → ŝ' on logged tuples."""
    logger.info("Training transition model — %d samples, %d epochs", len(states), epochs)

    s_t = torch.tensor(states, dtype=torch.float32, device=device)
    a_t = torch.tensor(actions, dtype=torch.long, device=device)
    ns_t = torch.tensor(next_states, dtype=torch.float32, device=device)

    n = len(states)
    losses: List[float] = []

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = agent.lyapunov.update_transition(
                s_t[idx], a_t[idx], ns_t[idx],
            )
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("  Transition Epoch %3d/%d — MSE=%.6f", epoch, epochs, avg_loss)

    logger.info("Transition model training complete — final MSE=%.6f", losses[-1])
    return losses


# ── Phase 3b: Lyapunov critic pre-training ──────────────────────────────────

def train_lyapunov_critic(
    agent: LyapunovPPOAgent,
    states: np.ndarray,
    constraint_costs: np.ndarray,
    next_states: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> List[float]:
    """Train the Lyapunov critic L_φ via TD loss on logged costs."""
    logger.info("Training Lyapunov critic — %d samples, %d epochs", len(states), epochs)

    s_t = torch.tensor(states, dtype=torch.float32, device=device)
    c_t = torch.tensor(constraint_costs, dtype=torch.float32, device=device)
    ns_t = torch.tensor(next_states, dtype=torch.float32, device=device)

    n = len(states)
    losses: List[float] = []

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = agent.lyapunov.update_critic(
                s_t[idx], c_t[idx], ns_t[idx],
            )
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("  Lyapunov Epoch %3d/%d — TD loss=%.6f", epoch, epochs, avg_loss)

    logger.info("Lyapunov critic training complete — final TD loss=%.6f", losses[-1])
    return losses


# ── Phase 3c: PPO with Lyapunov action masking ──────────────────────────────

def train_ppo_lyapunov(
    agent: LyapunovPPOAgent,
    traces: Dict[str, np.ndarray],
    epochs: int,
    rollout_length: int = 1000,
    minibatch_size: int = 256,
    ppo_epochs_per_rollout: int = 4,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: Path = Path("checkpoints/ppo_lyapunov"),
    checkpoint_interval: int = 20,
) -> Dict[str, List[float]]:
    """PPO training with Lyapunov action masking and safe-set tracking."""
    logger.info("Phase 3c: PPO+Lyapunov — %d epochs, rollout=%d", epochs, rollout_length)

    import pandas as pd

    # Create replay env.
    traces_path = checkpoint_dir / "replay_trace.parquet"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
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
    df.to_parquet(traces_path, engine="pyarrow", index=False)

    env = LatencyEnv(
        trace_path=str(traces_path),
        max_steps=rollout_length,
        latency_budget_ms=50.0,
    )

    metrics: Dict[str, List[float]] = {
        "reward": [],
        "constraint_cost": [],
        "violation_rate": [],
        "lyapunov_loss": [],
        "transition_loss": [],
        "safe_set_size": [],
        "mean_lyapunov_value": [],
        "policy_loss": [],
        "lambda": [],
    }

    for epoch in range(1, epochs + 1):
        states_list: List[np.ndarray] = []
        actions_list: List[int] = []
        rewards_list: List[float] = []
        values_list: List[float] = []
        log_probs_list: List[float] = []
        costs_list: List[float] = []
        dones_list: List[bool] = []
        next_states_list: List[np.ndarray] = []
        safe_set_sizes: List[int] = []
        lyap_values: List[float] = []

        obs, _ = env.reset()

        for step in range(rollout_length):
            state_tensor = torch.tensor(obs, dtype=torch.float32)
            action, log_prob, value, lyap_val = agent.select_action(state_tensor)

            # Track safe set size.
            safe_actions = agent.lyapunov.compute_safe_actions(state_tensor)
            safe_set_sizes.append(len(safe_actions))
            lyap_values.append(lyap_val)

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

        # Bootstrap value.
        with torch.no_grad():
            final_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            next_value = agent.critic(final_state).squeeze().item()

        returns, advantages = agent.compute_gae(
            rewards_list, values_list, dones_list, next_value,
        )

        states_t = torch.tensor(np.array(states_list), dtype=torch.float32)
        actions_t = torch.tensor(actions_list, dtype=torch.long)
        old_log_probs_t = torch.tensor(log_probs_list, dtype=torch.float32)
        costs_t = torch.tensor(costs_list, dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states_list), dtype=torch.float32)

        epoch_metrics: Dict[str, float] = {}
        for _ in range(ppo_epochs_per_rollout):
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

        mean_reward = float(np.mean(rewards_list))
        mean_cost = float(np.mean(costs_list))
        violation_rate = float(np.mean([1.0 if c > 0 else 0.0 for c in costs_list]))
        mean_safe_set = float(np.mean(safe_set_sizes))
        mean_lyap = float(np.mean(lyap_values))

        metrics["reward"].append(mean_reward)
        metrics["constraint_cost"].append(mean_cost)
        metrics["violation_rate"].append(violation_rate)
        metrics["lyapunov_loss"].append(epoch_metrics.get("lyapunov_loss", 0.0))
        metrics["transition_loss"].append(epoch_metrics.get("transition_loss", 0.0))
        metrics["safe_set_size"].append(mean_safe_set)
        metrics["mean_lyapunov_value"].append(mean_lyap)
        metrics["policy_loss"].append(epoch_metrics.get("policy_loss", 0.0))
        metrics["lambda"].append(epoch_metrics.get("lambda", 0.0))

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  Lyap-PPO Epoch %3d/%d — reward=%.4f  viol=%.2f%%  "
                "safe_set=%.1f  L(s)=%.4f  lyap_loss=%.6f  trans_loss=%.6f",
                epoch, epochs,
                mean_reward,
                violation_rate * 100,
                mean_safe_set,
                mean_lyap,
                epoch_metrics.get("lyapunov_loss", 0.0),
                epoch_metrics.get("transition_loss", 0.0),
            )

        if epoch % checkpoint_interval == 0:
            agent.save(checkpoint_dir)

    agent.save(checkpoint_dir)
    logger.info("Phase 3 complete — checkpoint saved to %s", checkpoint_dir)
    logger.info(
        "Final safe set size: %.1f  (target > 1.0)",
        metrics["safe_set_size"][-1] if metrics["safe_set_size"] else 0.0,
    )

    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Lyapunov-PPO agent from pre-trained PPO+Lagrangian.",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Directory containing Parquet trace files.",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        required=True,
        help="Directory with pre-trained PPO+Lagrangian checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="PPO+Lyapunov training epochs.")
    parser.add_argument(
        "--transition-epochs",
        type=int,
        default=50,
        help="Transition model pre-training epochs.",
    )
    parser.add_argument(
        "--critic-epochs",
        type=int,
        default=50,
        help="Lyapunov critic pre-training epochs.",
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=1000,
        help="Rollout length per epoch.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Device string.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/ppo_lyapunov"),
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

    # Build agent with Lyapunov enabled.
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
            "constraint_threshold": 0.01,
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

    # Load pre-trained PPO+Lagrangian weights.
    agent.load(args.pretrained)
    logger.info("Loaded pre-trained checkpoint from %s", args.pretrained)

    # Phase 3a: Train transition model on (s, a, s') tuples.
    train_transition_model(
        agent,
        traces["states"],
        traces["actions"],
        traces["next_states"],
        epochs=args.transition_epochs,
        device=device,
    )

    # Phase 3b: Train Lyapunov critic via TD.
    train_lyapunov_critic(
        agent,
        traces["states"],
        traces["constraint_costs"],
        traces["next_states"],
        epochs=args.critic_epochs,
        device=device,
    )

    # Phase 3c: PPO with Lyapunov action masking.
    metrics = train_ppo_lyapunov(
        agent,
        traces,
        epochs=args.epochs,
        rollout_length=args.rollout_length,
        minibatch_size=args.batch_size,
        device=device,
        checkpoint_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
