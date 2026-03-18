"""
src/latency_predictor.py
────────────────────────
Small MLP that predicts per-frame latency given (state, action).
Used by the conformal prediction layer to compute upper bounds.

Architecture
────────────
    Input : concat(state ∈ ℝ¹¹, one_hot(action) ∈ ℝ¹⁸) → ℝ²⁹
    Hidden: Linear(29, 32) → ReLU → Linear(32, 32) → ReLU
    Output: Linear(32, 1) → scalar latency (ms)
    Params: ~2,200

Public API
──────────
    LatencyPredictor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.torch_compat import torch_load_compat

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_STATE_DIM: int = 11
_NUM_ACTIONS: int = 18
_INPUT_DIM: int = _STATE_DIM + _NUM_ACTIONS  # 29


# ── Network ──────────────────────────────────────────────────────────────────

class _LatencyMLP(nn.Module):
    """Two-hidden-layer MLP: 29 → 32 → 32 → 1."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_INPUT_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── LatencyPredictor ─────────────────────────────────────────────────────────

class LatencyPredictor:
    """Predicts per-frame latency from (state, action) pairs.

    Parameters
    ----------
    config:
        ``conformal`` section from ``config/controller.yaml``.  Relevant keys::

            predictor_hidden_size: 32
            predictor_checkpoint: checkpoints/conformal/latency_predictor.pt
    device:
        Torch device (``cpu`` or ``cuda``).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device | None = None,
    ) -> None:
        self._device = device or torch.device("cpu")
        hidden = config.get("predictor_hidden_size", 32)
        self._checkpoint_path = Path(
            config.get("predictor_checkpoint", "checkpoints/conformal/latency_predictor.pt")
        )

        self._model = _LatencyMLP(hidden_size=hidden).to(self._device)
        self._model.eval()

        # Try loading existing checkpoint.
        if self._checkpoint_path.exists():
            self.load(self._checkpoint_path)
            logger.info(
                "Loaded latency predictor from %s (%d params)",
                self._checkpoint_path,
                sum(p.numel() for p in self._model.parameters()),
            )
        else:
            logger.info(
                "No latency predictor checkpoint found at %s; "
                "initialised with random weights (%d params)",
                self._checkpoint_path,
                sum(p.numel() for p in self._model.parameters()),
            )

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, state: torch.Tensor, action: int) -> float:
        """Return predicted latency (ms) for a single (state, action) pair.

        Parameters
        ----------
        state:
            Normalised state vector, shape ``(11,)`` or ``(1, 11)``.
        action:
            Action index in ``[0, 17]``.

        Returns
        -------
        float
            Predicted latency in milliseconds.
        """
        self._model.eval()
        x = self._build_input(state, action)
        return float(self._model(x).item())

    @torch.no_grad()
    def predict_batch(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Predict latencies for a batch.

        Parameters
        ----------
        states:
            ``(B, 11)`` normalised state batch.
        actions:
            ``(B,)`` integer action batch.

        Returns
        -------
        torch.Tensor
            ``(B,)`` predicted latencies.
        """
        self._model.eval()
        one_hot = torch.zeros(
            actions.shape[0], _NUM_ACTIONS, device=self._device
        )
        one_hot.scatter_(1, actions.unsqueeze(1).long(), 1.0)
        x = torch.cat([states, one_hot], dim=1)
        return self._model(x)

    # ── Training ─────────────────────────────────────────────────────────

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        latencies: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        val_fraction: float = 0.1,
    ) -> List[float]:
        """Train the predictor on calibration data via MSE loss.

        Parameters
        ----------
        states:
            ``(N, 11)`` normalised state array.
        actions:
            ``(N,)`` integer action array.
        latencies:
            ``(N,)`` observed latency array (ms).
        epochs:
            Number of training epochs.
        lr:
            Learning rate.
        batch_size:
            Mini-batch size.
        val_fraction:
            Fraction of data held out for validation logging.

        Returns
        -------
        List[float]
            Per-epoch training MSE losses.
        """
        n = len(states)
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val

        # Shuffle and split.
        perm = np.random.permutation(n)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        s_train = torch.tensor(states[train_idx], dtype=torch.float32, device=self._device)
        a_train = torch.tensor(actions[train_idx], dtype=torch.long, device=self._device)
        l_train = torch.tensor(latencies[train_idx], dtype=torch.float32, device=self._device)

        s_val = torch.tensor(states[val_idx], dtype=torch.float32, device=self._device)
        a_val = torch.tensor(actions[val_idx], dtype=torch.long, device=self._device)
        l_val = torch.tensor(latencies[val_idx], dtype=torch.float32, device=self._device)

        # One-hot encode actions.
        oh_train = torch.zeros(n_train, _NUM_ACTIONS, device=self._device)
        oh_train.scatter_(1, a_train.unsqueeze(1), 1.0)
        x_train = torch.cat([s_train, oh_train], dim=1)

        oh_val = torch.zeros(n_val, _NUM_ACTIONS, device=self._device)
        oh_val.scatter_(1, a_val.unsqueeze(1), 1.0)
        x_val = torch.cat([s_val, oh_val], dim=1)

        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self._model.train()
        epoch_losses: List[float] = []

        for epoch in range(epochs):
            perm_t = torch.randperm(n_train, device=self._device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                idx = perm_t[start : start + batch_size]
                pred = self._model(x_train[idx])
                loss = criterion(pred, l_train[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                with torch.no_grad():
                    val_pred = self._model(x_val)
                    val_loss = criterion(val_pred, l_val).item()
                logger.info(
                    "Epoch %3d/%d — train MSE: %.4f  val MSE: %.4f",
                    epoch + 1, epochs, avg_loss, val_loss,
                )

        self._model.eval()
        return epoch_losses

    # ── Checkpoint I/O ───────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """Save model weights to disk."""
        path = path or self._checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.info("Saved latency predictor to %s", path)

    def load(self, path: Optional[Path] = None) -> None:
        """Load model weights from disk."""
        path = path or self._checkpoint_path
        state_dict = torch_load_compat(
            path, map_location=self._device
        )
        self._model.load_state_dict(state_dict)
        self._model.eval()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _build_input(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """Concatenate state with one-hot action → (1, 29)."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self._device)
        one_hot = torch.zeros(1, _NUM_ACTIONS, device=self._device)
        one_hot[0, action] = 1.0
        return torch.cat([state, one_hot], dim=1)

    @property
    def model(self) -> nn.Module:
        """Access the underlying PyTorch module."""
        return self._model
