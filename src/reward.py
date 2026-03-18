"""
src/reward.py
─────────────
Shared reward computation for the Lyapunov-constrained RL pipeline.

Architecture formula (ARCHITECTURE.md §5):

    R(s_t, a_t) = 1.0 · Q_t  −  0.05 · ‖a_t − a_{t−1}‖₁

    Q_t = c̄_t · min(n_t, n_max) / n_max

This module is the **single source of truth** for reward computation,
action decoding, and factored L1 distance.  Both ``src/env.py`` (training)
and ``main.py`` (deployment) must import from here.

Public API
──────────
    compute_reward, decode_action, action_l1_distance
"""

from __future__ import annotations

from typing import Tuple


def decode_action(action_index: int) -> Tuple[int, int, int]:
    """Decode flat action index ∈ [0, 17] into factored components.

    Encoding: ``action_index = res_code * 6 + thr_code * 2 + seg_code``

    Returns
    -------
    (resolution_delta, threshold_delta, segmentation_flag)
        resolution_delta ∈ {-1, 0, +1}
        threshold_delta  ∈ {-1, 0, +1}
        segmentation_flag ∈ {0, 1}
    """
    seg = action_index % 2
    thr = (action_index // 2) % 3 - 1
    res = (action_index // 6) % 3 - 1
    return res, thr, seg


def action_l1_distance(a1: int, a2: int) -> float:
    """L1 distance between two actions in factored (res, thr, seg) space."""
    res1, thr1, seg1 = decode_action(a1)
    res2, thr2, seg2 = decode_action(a2)
    return float(abs(res1 - res2) + abs(thr1 - thr2) + abs(seg1 - seg2))


def compute_reward(
    mean_confidence: float,
    detection_count: int,
    curr_action: int,
    prev_action: int,
    n_max: int = 100,
    quality_weight: float = 1.0,
    churn_penalty: float = 0.05,
) -> float:
    """Compute per-step reward per ARCHITECTURE.md §5.

    R = quality_weight · Q_t  −  churn_penalty · ‖a_t − a_{t−1}‖₁

    Q_t = mean_confidence · min(detection_count, n_max) / n_max

    Parameters
    ----------
    mean_confidence : float
        Mean detection confidence c̄_t ∈ [0, 1].
    detection_count : int
        Number of detections n_t in the current frame.
    curr_action : int
        Current action index ∈ [0, 17].
    prev_action : int
        Previous action index ∈ [0, 17], or -1 if first step.
    n_max : int
        Maximum expected detection count (default 100, from config).
    quality_weight : float
        Weight on the quality proxy (default 1.0).
    churn_penalty : float
        Penalty coefficient for action churn (default 0.05).

    Returns
    -------
    float
        Scalar reward value.
    """
    quality = mean_confidence * min(detection_count, n_max) / n_max

    if prev_action < 0:
        churn = 0.0
    else:
        churn = action_l1_distance(curr_action, prev_action)

    return quality_weight * quality - churn_penalty * churn
