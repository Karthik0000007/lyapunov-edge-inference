"""
scripts/ablation.py
───────────────────
Ablation study automation: quantify the contribution of each safety layer.

Configuration matrix (2 × 2 × 2 = 8 variants):
    - Lyapunov action masking: on / off
    - Conformal prediction override: on / off
    - Rule-based fallback: on / off

Each variant is evaluated on all stress scenarios to measure the marginal
contribution of each safety layer.

Usage
─────
    python scripts/ablation.py \
        --checkpoint checkpoints/ppo_lyapunov \
        --traces data/telemetry.parquet \
        --seeds 3 \
        --output-dir results/ablation
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_lyapunov_ppo import LyapunovPPOAgent
from src.conformal import ConformalPredictor
from src.env import LatencyEnv
from src.state_features import ControllerState

logger = logging.getLogger(__name__)

_LATENCY_BUDGET_MS: float = 50.0
_NUM_ACTIONS: int = 18
_CONSERVATIVE_ACTION: int = 4


# ── Ablation controller ────────────────────────────────────────────────────


class AblationController:
    """Agent wrapper with toggleable safety layers."""

    def __init__(
        self,
        agent: LyapunovPPOAgent,
        lyapunov_enabled: bool = True,
        conformal_enabled: bool = True,
        fallback_enabled: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._agent = agent
        self._lyap_on = lyapunov_enabled
        self._conf_on = conformal_enabled
        self._fb_on = fallback_enabled
        self._device = device

        # Conformal predictor (only used if conformal_enabled).
        self._conformal: Optional[ConformalPredictor] = None
        if self._conf_on:
            conf_cfg = {
                "enabled": True,
                "alpha_target": 0.01,
                "alpha_lr": 0.005,
                "calibration_size": 5000,
                "predictor_hidden_size": 32,
            }
            self._conformal = ConformalPredictor(
                conf_cfg, latency_budget_ms=_LATENCY_BUDGET_MS, device=device
            )

        # Fallback state.
        self._consecutive_violations: int = 0
        self._clean_frames: int = 0
        self._fallback_active: bool = False
        self._violation_trigger: int = 3
        self._recovery_window: int = 50

        self._last_latency: float = 30.0

    @property
    def name(self) -> str:
        parts = ["PPO"]
        if self._lyap_on:
            parts.append("Lyap")
        if self._conf_on:
            parts.append("CP")
        if self._fb_on:
            parts.append("FB")
        return "+".join(parts)

    def reset(self) -> None:
        self._consecutive_violations = 0
        self._clean_frames = 0
        self._fallback_active = False
        self._last_latency = 30.0

    def select_action(self, obs: np.ndarray) -> int:
        state_tensor = torch.tensor(obs, dtype=torch.float32)

        # Layer 1: RL agent with optional Lyapunov masking.
        if self._lyap_on:
            action, _, _, _ = self._agent.select_action(state_tensor)
        else:
            # No Lyapunov masking — raw actor output.
            s = state_tensor.to(self._device).unsqueeze(0)
            with torch.no_grad():
                logits = self._agent.actor(s).squeeze(0)
                from torch.distributions import Categorical

                dist = Categorical(logits=logits)
                action = int(dist.sample().item())

        # Layer 2: Conformal override.
        if self._conf_on and self._conformal is not None:
            safe_actions = (
                self._agent.lyapunov.compute_safe_actions(state_tensor)
                if self._lyap_on
                else list(range(_NUM_ACTIONS))
            )
            action, _, _ = self._conformal.check_action(state_tensor, action, safe_actions)

        # Layer 3: Rule-based fallback.
        if self._fb_on:
            violated = self._last_latency > _LATENCY_BUDGET_MS
            if violated:
                self._consecutive_violations += 1
                self._clean_frames = 0
                if self._consecutive_violations >= self._violation_trigger:
                    self._fallback_active = True
            else:
                self._consecutive_violations = 0
                if self._fallback_active:
                    self._clean_frames += 1
                    if self._clean_frames >= self._recovery_window:
                        self._fallback_active = False
                        self._clean_frames = 0

            if self._fallback_active:
                action = _CONSERVATIVE_ACTION

        return action

    def observe_latency(self, latency_ms: float) -> None:
        self._last_latency = latency_ms


# ── Scenario runners (reuse from stress_test) ──────────────────────────────


def _run_ablation_variant(
    controller: AblationController,
    trace_path: str,
    seed: int,
    num_frames: int,
    scenario_name: str = "steady_state",
) -> Dict[str, Any]:
    """Run one ablation variant on one scenario/seed."""
    env = LatencyEnv(
        trace_path=trace_path,
        max_steps=num_frames,
        latency_budget_ms=_LATENCY_BUDGET_MS,
    )
    controller.reset()
    obs, _ = env.reset(seed=seed)

    latencies: List[float] = []
    violations: List[float] = []
    rewards: List[float] = []

    for _ in range(num_frames):
        action = controller.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        lat = info["latency_ms"]
        latencies.append(lat)
        violations.append(info["constraint_cost"])
        rewards.append(reward)
        controller.observe_latency(lat)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + 1000)

    lat_arr = np.array(latencies)

    return {
        "variant": controller.name,
        "scenario": scenario_name,
        "seed": seed,
        "lyapunov": controller._lyap_on,
        "conformal": controller._conf_on,
        "fallback": controller._fb_on,
        "p50_latency_ms": float(np.percentile(lat_arr, 50)),
        "p95_latency_ms": float(np.percentile(lat_arr, 95)),
        "p99_latency_ms": float(np.percentile(lat_arr, 99)),
        "mean_latency_ms": float(np.mean(lat_arr)),
        "violation_rate": float(np.mean(violations)),
        "mean_reward": float(np.mean(rewards)),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation study on safety layers.",
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
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds.")
    parser.add_argument("--frames", type=int, default=2000, help="Frames per run.")
    parser.add_argument("--device", type=str, default="cpu", help="Device string.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ablation"),
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
        "agent": {"checkpoint_dir": str(args.checkpoint)},
    }
    agent = LyapunovPPOAgent(config, device=device)
    agent.load(args.checkpoint)
    logger.info("Loaded agent from %s", args.checkpoint)

    # Build configuration matrix: Lyapunov × Conformal × Fallback.
    variants: List[AblationController] = []
    for lyap, conf, fb in itertools.product([True, False], repeat=3):
        variants.append(
            AblationController(
                agent,
                lyapunov_enabled=lyap,
                conformal_enabled=conf,
                fallback_enabled=fb,
                device=device,
            )
        )

    # Scenarios to test each variant on.
    scenarios = ["steady_state"]

    logger.info(
        "Running %d variants × %d scenarios × %d seeds × %d frames",
        len(variants),
        len(scenarios),
        args.seeds,
        args.frames,
    )

    all_results: List[Dict[str, Any]] = []

    for variant in variants:
        for scenario in scenarios:
            for seed_idx in range(args.seeds):
                seed = seed_idx * 42
                logger.info(
                    "  %s | %s | seed %d/%d", variant.name, scenario, seed_idx + 1, args.seeds
                )
                metrics = _run_ablation_variant(variant, args.traces, seed, args.frames, scenario)
                all_results.append(metrics)
                logger.info(
                    "    P99=%.2f ms  viol=%.2f%%  reward=%.4f",
                    metrics["p99_latency_ms"],
                    metrics["violation_rate"] * 100,
                    metrics["mean_reward"],
                )

    # Write per-run CSV.
    csv_path = args.output_dir / "ablation_per_run.csv"
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    logger.info("Per-run results → %s", csv_path)

    # Aggregate per variant.
    variant_names = list(dict.fromkeys(r["variant"] for r in all_results))
    agg_rows: List[Dict[str, Any]] = []
    for vname in variant_names:
        runs = [r for r in all_results if r["variant"] == vname]
        row: Dict[str, Any] = {
            "variant": vname,
            "lyapunov": runs[0]["lyapunov"],
            "conformal": runs[0]["conformal"],
            "fallback": runs[0]["fallback"],
        }
        for key in [
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "mean_latency_ms",
            "violation_rate",
            "mean_reward",
        ]:
            vals = [r[key] for r in runs]
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_std"] = float(np.std(vals))
        agg_rows.append(row)

    agg_path = args.output_dir / "ablation_summary.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    logger.info("Ablation summary → %s", agg_path)

    # ── Contribution analysis ───────────────────────────────────────────
    # Compare full system (all on) vs. each layer removed.
    logger.info("=" * 80)
    logger.info("ABLATION STUDY — LAYER CONTRIBUTION ANALYSIS")
    logger.info("-" * 80)
    logger.info("%-25s %8s %8s %8s %8s", "Variant", "P99", "Viol%", "Reward", "Delta P99")

    full_system = next(
        (r for r in agg_rows if r["lyapunov"] and r["conformal"] and r["fallback"]), None
    )
    full_p99 = full_system["p99_latency_ms_mean"] if full_system else 0.0

    for row in agg_rows:
        delta = row["p99_latency_ms_mean"] - full_p99
        logger.info(
            "%-25s %8.2f %7.2f%% %8.4f %+8.2f",
            row["variant"],
            row["p99_latency_ms_mean"],
            row["violation_rate_mean"] * 100,
            row["mean_reward_mean"],
            delta,
        )

    logger.info("=" * 80)

    # LaTeX ablation table.
    latex_path = args.output_dir / "ablation_table.tex"
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation study: contribution of each safety layer}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\toprule\n")
        f.write(
            "Variant & Lyap. & CP & FB & P99 (ms) & Viol. (\\%) & Reward & " "$\\Delta$P99 \\\\\n"
        )
        f.write("\\midrule\n")
        for row in agg_rows:
            delta = row["p99_latency_ms_mean"] - full_p99
            name = row["variant"].replace("+", "{+}")
            f.write(
                "%s & %s & %s & %s & %.1f & %.1f & %.3f & %+.1f \\\\\n"
                % (
                    name,
                    "\\cmark" if row["lyapunov"] else "\\xmark",
                    "\\cmark" if row["conformal"] else "\\xmark",
                    "\\cmark" if row["fallback"] else "\\xmark",
                    row["p99_latency_ms_mean"],
                    row["violation_rate_mean"] * 100,
                    row["mean_reward_mean"],
                    delta,
                )
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    logger.info("LaTeX table → %s", latex_path)


if __name__ == "__main__":
    main()
