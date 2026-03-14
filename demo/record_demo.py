"""
demo/record_demo.py
───────────────────
Record a demo video that showcases the adaptive inference pipeline:
frame-by-frame processing with annotation overlays and controller
decision text.

Usage
─────
    python -m demo.record_demo --input video.mp4 --output demo/demo_video.mp4
    python -m demo.record_demo --input video.mp4 --checkpoint checkpoints/ppo_lyapunov/
    python -m demo.record_demo --input video.mp4 --side-by-side
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root to path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.controller import AdaptiveController
from src.state_features import ControllerAction, ControllerState
from src.utils import load_config

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_OUTPUT = "demo/demo_video.mp4"
_RESOLUTION_MAP = {0: 320, 1: 480, 2: 640}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_COLOR = (0, 255, 0)  # Green
_FONT_THICKNESS = 1
_BG_COLOR = (0, 0, 0)
_OVERLAY_ALPHA = 0.65


# ── Annotation helpers ──────────────────────────────────────────────────────


def _draw_overlay_text(frame: np.ndarray, lines: list[str], origin: tuple[int, int]) -> np.ndarray:
    """Draw text lines with a semi-transparent background box."""
    overlay = frame.copy()
    x, y = origin
    line_height = 20
    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        max_width = max(max_width, w)

    # Background rectangle.
    pad = 6
    cv2.rectangle(
        overlay,
        (x - pad, y - line_height - pad),
        (x + max_width + pad, y + line_height * len(lines) + pad),
        _BG_COLOR,
        cv2.FILLED,
    )
    frame = cv2.addWeighted(overlay, _OVERLAY_ALPHA, frame, 1 - _OVERLAY_ALPHA, 0)

    # Text lines.
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_height),
            _FONT,
            _FONT_SCALE,
            _FONT_COLOR,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return frame


def _build_synthetic_state(
    frame_idx: int,
    fps: float,
    resolution_index: int,
    threshold_index: int,
    segmentation_enabled: bool,
) -> ControllerState:
    """Build a ControllerState from frame metadata (simulated telemetry)."""
    rng = np.random.default_rng(frame_idx)
    latency = 20.0 + 15.0 * rng.standard_normal()
    latency = float(np.clip(latency, 5.0, 100.0))
    return ControllerState(
        last_latency_ms=latency,
        mean_latency_ms=latency * 0.95,
        p99_latency_ms=latency * 1.3,
        detection_count=int(rng.poisson(3)),
        mean_confidence=float(rng.beta(5, 2)),
        defect_area_ratio=float(rng.beta(2, 20)),
        resolution_index=resolution_index,
        threshold_index=threshold_index,
        segmentation_enabled=int(segmentation_enabled),
        gpu_utilization=50.0 + 20.0 * float(rng.random()),
        gpu_temperature_norm=50.0 + 15.0 * float(rng.random()),
    )


def _make_default_config() -> dict:
    """Create a minimal controller config for demo mode."""
    config_path = PROJECT_ROOT / "config" / "controller.yaml"
    if config_path.exists():
        return load_config(config_path)
    return {
        "agent": {"decision_frequency": 5, "checkpoint_dir": "checkpoints/ppo_lyapunov/"},
        "ppo": {
            "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
            "entropy_coeff": 0.01, "value_loss_coeff": 0.5,
            "max_grad_norm": 0.5, "hidden_size": 64,
        },
        "lagrangian": {"lambda_init": 0.1, "lambda_lr": 0.01, "constraint_threshold": 0.01},
        "lyapunov": {"enabled": True, "critic_lr": 3e-4, "drift_tolerance": 0.0},
        "conformal": {
            "enabled": False, "alpha_target": 0.01, "alpha_lr": 0.005,
            "calibration_size": 5000, "predictor_hidden_size": 32,
        },
        "fallback": {"enabled": True, "consecutive_violations": 3, "recovery_window": 50},
    }


# ── Main recording loop ─────────────────────────────────────────────────────


def record_demo(
    input_path: str,
    output_path: str = _DEFAULT_OUTPUT,
    checkpoint_dir: str | None = None,
    side_by_side: bool = False,
) -> Path:
    """Record an annotated demo video from an input video file.

    Parameters
    ----------
    input_path:
        Path to input video file.
    output_path:
        Path for the output MP4 file.
    checkpoint_dir:
        Optional path to agent checkpoint directory.
    side_by_side:
        If True, render original and annotated frames side by side.

    Returns
    -------
    Path
        Path to the output video file.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Input video: %s — %dx%d @ %.1f fps, %d frames",
        input_path, width, height, fps, total_frames,
    )

    # Output dimensions.
    out_width = width * 2 if side_by_side else width
    out_height = height

    # Create output directory.
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer for {output_path}")

    # Initialise controller.
    config = _make_default_config()
    if checkpoint_dir:
        config["agent"]["checkpoint_dir"] = checkpoint_dir
    controller = AdaptiveController(config, device=torch.device("cpu"))

    resolution_index = 2
    threshold_index = 0
    segmentation_enabled = True

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Build synthetic state from frame metadata.
        state = _build_synthetic_state(
            frame_idx, fps, resolution_index, threshold_index, segmentation_enabled,
        )

        # Run controller.
        action = controller.step(state, observed_latency=state.last_latency_ms)
        decision = controller.last_decision

        # Update running config.
        resolution_index = controller.resolution_index
        threshold_index = controller.threshold_index
        segmentation_enabled = controller.segmentation_enabled

        # Annotate frame.
        annotated = frame.copy()
        overlay_lines = [
            f"Frame: {frame_idx}/{total_frames}",
            f"Action: {action.action_index}  "
            f"(res={action.resolution_delta:+d}, thr={action.threshold_delta:+d}, "
            f"seg={'ON' if action.segmentation_enabled else 'OFF'})",
            f"Resolution: {_RESOLUTION_MAP.get(resolution_index, '?')}px",
            f"Latency: {state.last_latency_ms:.1f} ms",
        ]
        if decision is not None:
            overlay_lines.extend([
                f"Lyapunov L(s): {decision.lyapunov_value:.3f}",
                f"Conformal U_t: {decision.conformal_bound:.1f} ms"
                f"{'  [OVERRIDE]' if decision.conformal_overridden else ''}",
                f"Fallback: {'ACTIVE' if decision.fallback_active else 'inactive'}",
            ])

        annotated = _draw_overlay_text(annotated, overlay_lines, (10, 25))

        # Write frame.
        if side_by_side:
            combined = np.hstack([frame, annotated])
            writer.write(combined)
        else:
            writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            logger.info("Processed %d / %d frames", frame_idx, total_frames)

    cap.release()
    writer.release()
    logger.info("Demo video saved to %s (%d frames)", output_path, frame_idx)
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record an annotated demo video from the adaptive inference pipeline.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output", "-o",
        default=_DEFAULT_OUTPUT,
        help=f"Output MP4 path (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Path to agent checkpoint directory.",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Render original and annotated frames side by side.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    output = record_demo(
        input_path=args.input,
        output_path=args.output,
        checkpoint_dir=args.checkpoint,
        side_by_side=args.side_by_side,
    )
    print(f"Demo video written to: {output}")


if __name__ == "__main__":
    main()
