"""
main.py
-------
Pipeline orchestrator — main entry point for the Lyapunov-constrained
RL edge inference system.

Responsibilities:
    1. Startup: load configs, TensorRT engines, RL checkpoints, conformal
       state, camera thread, telemetry logger, drift monitor, dashboard.
    2. Inference loop: dequeue → preprocess → detect → controller → segment
       → post-process → telemetry → optional online PPO update.
    3. Shutdown: signal-based graceful teardown with resource cleanup.

Usage::

    python main.py --config config/pipeline.yaml --source 0
    python main.py --config config/pipeline.yaml --agent checkpoints/ppo_lyapunov/ --no-dashboard
"""

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np
import torch

from src.camera import CameraCapture
from src.controller import AdaptiveController
from src.detection import DetectionEngine
from src.drift import DriftMonitor
from src.monitoring import GPUMonitor, MetricsWindow, TelemetryLogger
from src.preprocess import Preprocessor
from src.reward import compute_reward
from src.segmentation import SegmentationEngine
from src.state_features import (
    ControllerAction,
    ControllerState,
    Detection,
    TelemetryRecord,
    Transition,
)
from src.telemetry import FrameTimer
from src.utils import (
    compute_sha256,
    load_all_configs,
    setup_device,
    setup_logging,
)

logger = logging.getLogger("pipeline")

# ── Resolution lookup table ──────────────────────────────────────────────────

_RESOLUTIONS: List[int] = [320, 480, 640]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Lyapunov-constrained RL edge inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--controller-config",
        type=str,
        default="config/controller.yaml",
        help="Path to controller YAML config.",
    )
    parser.add_argument(
        "--deployment-config",
        type=str,
        default=None,
        help="Optional deployment overrides YAML.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Override checkpoint directory for RL agent.",
    )
    parser.add_argument(
        "--conformal",
        type=str,
        default=None,
        help="Override path to conformal predictor state file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override camera source (device index or file path).",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable Streamlit dashboard subprocess.",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Path to save recorded video (e.g., demo/output.mp4). If not specified, no recording.",
    )
    parser.add_argument(
        "--slowdown",
        type=float,
        default=1.0,
        help="Slowdown factor for recorded video (e.g., 2.0 = half speed, 3.0 = third speed).",
    )
    return parser.parse_args()


# ── Startup helpers ──────────────────────────────────────────────────────────

def _load_checkpoints_with_integrity(agent_dir: Path) -> None:
    """Log SHA-256 digests for all checkpoint files in *agent_dir*."""
    expected_files = [
        "actor.pt", "critic.pt", "lagrangian.pt",
        "lyapunov_critic.pt", "transition_model.pt",
    ]
    for fname in expected_files:
        fpath = agent_dir / fname
        if fpath.exists():
            digest = compute_sha256(fpath)
            logger.info("Checkpoint %s  SHA-256=%s", fpath.name, digest[:16])
        else:
            logger.warning("Checkpoint file missing: %s", fpath)


def _launch_dashboard(cfg: Dict[str, Any]) -> Optional[subprocess.Popen]:
    """Launch Streamlit dashboard as a subprocess (Windows-safe)."""
    dashboard_cfg = cfg.get("dashboard", {})
    host = dashboard_cfg.get("host", "127.0.0.1")
    port = str(dashboard_cfg.get("port", 8501))
    theme = dashboard_cfg.get("theme", "dark")

    # Search common dashboard script locations.
    candidates = [
        Path("app/dashboard.py"),
        Path("dashboard.py"),
        Path("src/dashboard.py"),
    ]
    dashboard_path: Optional[Path] = None
    for p in candidates:
        if p.exists():
            dashboard_path = p
            break

    if dashboard_path is None:
        logger.warning(
            "Dashboard script not found in %s — skipping launch",
            [str(c) for c in candidates],
        )
        return None

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.address", host,
        "--server.port", port,
        "--theme.base", theme,
        "--server.headless", "true",
    ]
    try:
        # On Windows, use CREATE_NEW_PROCESS_GROUP so the dashboard does
        # not receive the same Ctrl-C as the main process.
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32" else 0
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        logger.info(
            "Dashboard launched  pid=%d  http://%s:%s", proc.pid, host, port,
        )
        return proc
    except Exception:
        logger.warning("Failed to launch dashboard subprocess", exc_info=True)
        return None


# ── Frame annotation overlay ────────────────────────────────────────────────

def _annotate_frame(
    image: np.ndarray,
    detections: List[Detection],
    masks: Optional[List[np.ndarray]],
    latency_ms: float,
    action: ControllerAction,
    resolution: int,
) -> np.ndarray:
    """Draw bounding boxes, masks, and HUD overlay on a copy of *image*."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # Draw detection bounding boxes.
    for det in detections:
        x1 = int(det.bbox[0] * w)
        y1 = int(det.bbox[1] * h)
        x2 = int(det.bbox[2] * w)
        y2 = int(det.bbox[3] * h)
        colour = (0, 255, 0) if det.confidence >= 0.3 else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            vis, label, (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1,
        )

    # Overlay segmentation masks with transparency.
    if masks:
        overlay = vis.copy()
        for mask in masks:
            if mask is not None and mask.shape[:2] == vis.shape[:2]:
                overlay[mask > 0] = (0, 0, 200)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

    # HUD text.
    seg_str = "ON" if action.segmentation_enabled else "OFF"
    hud_lines = [
        f"Latency: {latency_ms:.1f} ms",
        f"Res: {resolution}  Act: {action.action_index}  Seg: {seg_str}",
        f"Detections: {len(detections)}",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(
            vis, line, (8, 18 + i * 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )

    # Status overlay (bottom-right): green GOOD when no confident detection, red DEFECT when any detection >= 0.3
    is_defect = any(det.confidence >= 0.3 for det in detections)
    status_text = "DEFECT" if is_defect else "GOOD"
    status_color = (0, 0, 255) if is_defect else (0, 200, 0)

    status_label = f"Status: {status_text}"
    (tw, th), _ = cv2.getTextSize(status_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad = 8
    x1 = w - tw - pad - 10
    y1 = h - th - pad - 10  # Bottom instead of top
    x2 = w - 10
    y2 = h - 10  # Bottom instead of top
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(
        vis, status_label, (x1 + 4, y2 - 4),  # Adjusted y position
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1,
    )

    return vis


# ── Online training buffer ───────────────────────────────────────────────────

class _OnlineBuffer:
    """Ring buffer for online PPO transitions."""

    def __init__(self, capacity: int = 4096) -> None:
        self._buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def __len__(self) -> int:
        return len(self._buf)

    def drain(self) -> List[Transition]:
        """Return all transitions and clear the buffer."""
        items = list(self._buf)
        self._buf.clear()
        return items


def _run_online_update(
    agent: Any,
    buffer: _OnlineBuffer,
    device: torch.device,
) -> None:
    """Execute one online PPO update from buffered transitions."""
    transitions = buffer.drain()
    if not transitions:
        return

    states = torch.tensor(
        np.array([t.state for t in transitions]), dtype=torch.float32,
    )
    actions = torch.tensor(
        [t.action for t in transitions], dtype=torch.long,
    )
    old_log_probs = torch.tensor(
        [t.log_prob for t in transitions], dtype=torch.float32,
    )
    rewards = [t.reward for t in transitions]
    values = [t.value for t in transitions]
    dones = [t.done for t in transitions]
    constraint_costs = torch.tensor(
        [t.constraint_cost for t in transitions], dtype=torch.float32,
    )
    next_states = torch.tensor(
        np.array([t.next_state for t in transitions]), dtype=torch.float32,
    )

    # Compute GAE.
    next_value = values[-1] if values else 0.0
    returns, advantages = agent.compute_gae(rewards, values, dones, next_value)

    # PPO update.
    metrics = agent.update(
        states, actions, old_log_probs,
        returns, advantages, constraint_costs, next_states,
    )
    logger.info(
        "Online PPO update — loss=%.4f  lambda=%.4f  lyap=%.4f",
        metrics.get("total_loss", 0.0),
        metrics.get("lambda", 0.0),
        metrics.get("lyapunov_loss", 0.0),
    )


# ── Main pipeline ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. Load configs ─────────────────────────────────────────────────
    cfg = load_all_configs(
        pipeline_path=args.config,
        controller_path=args.controller_config,
        deployment_path=args.deployment_config,
    )

    log_level = cfg.get("telemetry", {}).get("log_level", "INFO")
    setup_logging(level=log_level, logger_name=None)
    logger.info("Configuration loaded")

    # ── 2. Device setup ─────────────────────────────────────────────────
    device = setup_device()
    logger.info("Device: %s", device)

    # ── 3. TensorRT engines ─────────────────────────────────────────────
    det_cfg = cfg.get("detection", {})
    seg_cfg = cfg.get("segmentation", {})

    detection_engine: Optional[DetectionEngine] = None
    segmentation_engine: Optional[SegmentationEngine] = None

    try:
        detection_engine = DetectionEngine(det_cfg)
        logger.info("Detection TensorRT engines loaded")
    except (RuntimeError, FileNotFoundError) as exc:
        logger.warning(
            "Detection engine load failed (%s) — running in stub mode", exc,
        )

    try:
        segmentation_engine = SegmentationEngine(seg_cfg)
        logger.info("Segmentation TensorRT engine loaded")
    except (RuntimeError, FileNotFoundError) as exc:
        logger.warning(
            "Segmentation engine load failed (%s) — detection-only mode", exc,
        )

    # ── 4. RL agent + controller ────────────────────────────────────────
    controller_cfg = {
        k: cfg[k]
        for k in (
            "agent", "ppo", "lagrangian", "lyapunov",
            "conformal", "fallback", "reward", "online_training",
        )
        if k in cfg
    }

    # Override checkpoint dir if CLI flag provided.
    if args.agent:
        controller_cfg.setdefault("agent", {})["checkpoint_dir"] = args.agent

    controller = AdaptiveController(controller_cfg, device=device)

    # Load RL checkpoints with SHA-256 integrity logging.
    agent_dir = Path(
        controller_cfg.get("agent", {}).get(
            "checkpoint_dir", "checkpoints/ppo_lyapunov/",
        )
    )
    if agent_dir.exists():
        _load_checkpoints_with_integrity(agent_dir)
        controller.agent.load(agent_dir)
        logger.info("RL agent checkpoints loaded from %s", agent_dir)
    else:
        logger.warning(
            "Agent checkpoint dir %s not found — using random weights",
            agent_dir,
        )

    # ── 5. Conformal predictor state ────────────────────────────────────
    conformal_state_path: Optional[Path] = None
    if args.conformal:
        conformal_state_path = Path(args.conformal)
    else:
        default_conformal = Path("checkpoints/conformal/conformal_state.pt")
        if default_conformal.exists():
            conformal_state_path = default_conformal

    if conformal_state_path and conformal_state_path.exists():
        controller.conformal.load_state(conformal_state_path)
        logger.info("Conformal state loaded from %s", conformal_state_path)

    # ── 6. Preprocessor ─────────────────────────────────────────────────
    preprocess_cfg = cfg.get("preprocessing", {})
    preprocessor = Preprocessor(preprocess_cfg)

    # ── 7. Camera thread ────────────────────────────────────────────────
    cam_cfg = cfg.get("camera", {})
    cam_source = (
        args.source if args.source is not None
        else cam_cfg.get("source", 0)
    )
    try:
        cam_source = int(cam_source)
    except (ValueError, TypeError):
        pass

    camera = CameraCapture(
        source=cam_source,
        fps=cam_cfg.get("fps", 30),
        queue_size=cam_cfg.get("queue_size", 10),
    )

    # ── 7b. Video recording setup ───────────────────────────────────────
    video_writer: Optional[cv2.VideoWriter] = None
    record_path: Optional[Path] = None
    slowdown_factor: float = args.slowdown if hasattr(args, 'slowdown') else 1.0
    if args.record:
        record_path = Path(args.record)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if slowdown_factor > 1.0:
            logger.info("Video recording will be saved to %s (slowed down %.1fx)", record_path, slowdown_factor)
        else:
            logger.info("Video recording will be saved to %s", record_path)

    # ── 8. Telemetry logger ─────────────────────────────────────────────
    tel_cfg = cfg.get("telemetry", {})
    telemetry_logger = TelemetryLogger(
        output_dir=tel_cfg.get("output_dir", "traces/"),
        rotation_frames=tel_cfg.get("rotation_frames", 100_000),
        window_size=cfg.get("latency", {}).get("window_size", 50),
    )

    # ── 9. GPU monitor ──────────────────────────────────────────────────
    gpu_monitor = GPUMonitor(query_interval=5)

    # ── 10. Drift monitor ───────────────────────────────────────────────
    drift_monitor = DriftMonitor()

    # ── 11. Latency statistics window ───────────────────────────────────
    latency_window = MetricsWindow(
        window_size=cfg.get("latency", {}).get("window_size", 50),
    )
    latency_budget_ms: float = cfg.get("latency", {}).get("budget_ms", 50.0)

    # ── 12. Online training setup ───────────────────────────────────────
    online_cfg = cfg.get("online_training", {})
    online_enabled: bool = online_cfg.get("enabled", False)
    online_update_interval: int = online_cfg.get("update_interval", 1000)
    online_min_buffer: int = online_cfg.get("min_buffer_size", 2000)
    online_buffer = _OnlineBuffer(
        capacity=max(online_min_buffer * 2, 4096),
    )

    if online_enabled:
        logger.info(
            "Online training ENABLED  interval=%d  min_buffer=%d",
            online_update_interval, online_min_buffer,
        )

    # ── 13. Dashboard subprocess ────────────────────────────────────────
    dashboard_proc: Optional[subprocess.Popen] = None
    if not args.no_dashboard:
        dashboard_proc = _launch_dashboard(cfg)

    # ── 14. Shutdown event + signal handlers ────────────────────────────
    shutdown_event = threading.Event()

    def _signal_handler(signum: int, _frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating graceful shutdown", sig_name)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    # SIGTERM does not exist on Windows; use SIGBREAK instead.
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, _signal_handler)
    else:
        signal.signal(signal.SIGTERM, _signal_handler)

    # ── 15. Start camera ────────────────────────────────────────────────
    camera.start()
    logger.info("Pipeline startup complete — entering inference loop")

    # ── Pipeline state ──────────────────────────────────────────────────
    conf_steps = det_cfg.get("confidence_steps", [0.0, 0.1, 0.2])
    conf_base = det_cfg.get("confidence_base", 0.25)
    n_max: int = det_cfg.get("max_detections", 100)

    reward_cfg = cfg.get("reward", {})
    reward_quality_weight: float = reward_cfg.get("quality_weight", 1.0)
    reward_churn_penalty: float = reward_cfg.get("churn_penalty", 0.05)

    prev_action_idx: int = 8  # default no-op
    prev_detections: List[Detection] = []
    frame_count: int = 0
    last_latency_ms: float = 0.0

    held_action = ControllerAction.from_index(8)

    # ═══════════════════════════════════════════════════════════════════
    #  MAIN INFERENCE LOOP
    # ═══════════════════════════════════════════════════════════════════

    try:
        while not shutdown_event.is_set():
            # ── Dequeue frame ────────────────────────────────────────
            frame = camera.get_frame(timeout=1.0)
            if frame is None:
                if not camera.is_alive:
                    logger.info("Camera thread ended — exiting loop")
                    break
                continue

            timer = FrameTimer()
            frame_count += 1

            # ── GPU metrics ──────────────────────────────────────────
            gpu_monitor.update()

            # ── Preprocess ───────────────────────────────────────────
            active_res = _RESOLUTIONS[controller.resolution_index]
            with timer.stage("preprocess"):
                preprocessed = preprocessor.process(
                    frame.raw_image,
                    target_resolution=active_res,
                    prior_detections=prev_detections,
                )
            frame.preprocessed = preprocessed
            frame.timestamp_preprocessed = float(time.perf_counter_ns())

            # ── Detect ───────────────────────────────────────────────
            with timer.stage("detect"):
                if detection_engine is not None:
                    detection_engine.set_resolution_index(
                        controller.resolution_index,
                    )
                    detections = detection_engine.detect(
                        preprocessed,
                        threshold_index=controller.threshold_index,
                    )
                else:
                    detections = []
            frame.detections = detections
            frame.timestamp_detected = float(time.perf_counter_ns())

            # ── Controller decision (every k-th frame) ───────────────
            mean_conf = 0.0
            defect_area = 0.0
            if detections:
                mean_conf = (
                    sum(d.confidence for d in detections) / len(detections)
                )
                defect_area = sum(
                    (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
                    for d in detections
                )

            state = ControllerState(
                last_latency_ms=last_latency_ms,
                mean_latency_ms=latency_window.mean,
                p99_latency_ms=latency_window.p99,
                detection_count=len(detections),
                mean_confidence=mean_conf,
                defect_area_ratio=min(defect_area, 1.0),
                resolution_index=controller.resolution_index,
                threshold_index=controller.threshold_index,
                segmentation_enabled=int(controller.segmentation_enabled),
                gpu_utilization=gpu_monitor.utilization,
                gpu_temperature_norm=gpu_monitor.temperature,
            )

            action = controller.step(
                state, last_latency_ms, latency_budget_ms,
            )
            held_action = action

            # ── Conditional segmentation ─────────────────────────────
            masks = None
            seg_enabled = controller.segmentation_enabled
            with timer.stage("segment"):
                if (
                    seg_enabled
                    and segmentation_engine is not None
                    and detections
                ):
                    segmentation_engine.enabled = True
                    masks = segmentation_engine.segment(
                        frame.raw_image, detections,
                    )
                elif segmentation_engine is not None:
                    segmentation_engine.enabled = False

            frame.masks = masks
            frame.stage2_executed = masks is not None
            frame.timestamp_segmented = float(time.perf_counter_ns())

            # ── Post-process + annotate ──────────────────────────────
            with timer.stage("postprocess"):
                annotated = _annotate_frame(
                    frame.raw_image,
                    detections,
                    masks,
                    timer.total_ms,
                    held_action,
                    active_res,
                )

            frame.timestamp_completed = float(time.perf_counter_ns())

            # ── Video recording ──────────────────────────────────────
            if record_path is not None:
                # Lazy initialization of VideoWriter on first frame
                if video_writer is None:
                    h, w = annotated.shape[:2]
                    input_fps = cam_cfg.get("fps", 30)
                    # Reduce output FPS to slow down video
                    output_fps = max(10, int(input_fps / slowdown_factor))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        str(record_path), fourcc, output_fps, (w, h)
                    )
                    logger.info("Video writer initialized: %dx%d @ %d fps (%.1fx slowdown)",
                               w, h, output_fps, slowdown_factor)

                # Write annotated frame
                if video_writer is not None and video_writer.isOpened():
                    video_writer.write(annotated)

            # ── Latency tracking ─────────────────────────────────────
            total_latency_ms = timer.total_ms
            last_latency_ms = total_latency_ms
            latency_window.push(total_latency_ms)

            frame.latency_ms = total_latency_ms
            frame.controller_action = held_action.action_index
            frame.active_resolution = (active_res, active_res)

            # ── Drift monitor ────────────────────────────────────────
            ks_p, drift_alert = drift_monitor.update(
                frame.raw_image, mean_conf,
            )

            # ── Conformal online update ──────────────────────────────
            state_tensor = state.to_tensor()
            controller.conformal.update(
                state_tensor, held_action.action_index, total_latency_ms,
            )

            # ── Reward / constraint ──────────────────────────────────
            constraint_cost = (
                1.0 if total_latency_ms > latency_budget_ms else 0.0
            )
            reward = compute_reward(
                mean_confidence=mean_conf,
                detection_count=len(detections),
                curr_action=held_action.action_index,
                prev_action=prev_action_idx,
                n_max=n_max,
                quality_weight=reward_quality_weight,
                churn_penalty=reward_churn_penalty,
            )

            # ── Controller introspection ─────────────────────────────
            lyap_val = 0.0
            conformal_bound = 0.0
            decision = controller.last_decision
            if decision is not None:
                lyap_val = decision.lyapunov_value
                conformal_bound = decision.conformal_bound

            # ── Telemetry record ─────────────────────────────────────
            record = TelemetryRecord(
                frame_id=frame.frame_id,
                timestamp=time.time(),
                latency_ms=total_latency_ms,
                latency_preprocess_ms=timer.preprocess_ms,
                latency_detect_ms=timer.detect_ms,
                latency_segment_ms=timer.segment_ms,
                latency_postprocess_ms=timer.postprocess_ms,
                detection_count=len(detections),
                mean_confidence=mean_conf,
                defect_area_ratio=min(defect_area, 1.0),
                controller_action=held_action.action_index,
                resolution_active=active_res,
                segmentation_active=seg_enabled,
                threshold_active=conf_base + conf_steps[
                    min(controller.threshold_index, len(conf_steps) - 1)
                ],
                gpu_util_percent=gpu_monitor.utilization,
                gpu_temp_celsius=gpu_monitor.temperature,
                gpu_memory_used_mb=gpu_monitor.memory_used_mb,
                conformal_upper_bound_ms=conformal_bound,
                conformal_alpha=controller.conformal.alpha,
                ks_p_value=ks_p,
                drift_alert=drift_alert,
                lyapunov_value=lyap_val,
                constraint_cost=constraint_cost,
                reward=reward,
            )
            telemetry_logger.log(record)

            # ── Online training buffer ───────────────────────────────
            if online_enabled:
                transition = Transition(
                    state=state_tensor.numpy(),
                    action=held_action.action_index,
                    reward=reward,
                    constraint_cost=constraint_cost,
                    next_state=state_tensor.numpy(),
                    done=False,
                    log_prob=0.0,
                    value=lyap_val,
                    lyapunov_value=lyap_val,
                )
                online_buffer.push(transition)

                # Gated PPO update every N steps when buffer is full.
                if (
                    frame_count % online_update_interval == 0
                    and len(online_buffer) >= online_min_buffer
                ):
                    _run_online_update(
                        controller.agent, online_buffer, device,
                    )

            # ── Periodic status logging ──────────────────────────────
            if frame_count % 100 == 0:
                logger.info(
                    "Frame %d  latency=%.1f ms  mean=%.1f ms  "
                    "p99=%.1f ms  dets=%d  res=%d  seg=%s  drift=%s",
                    frame_count,
                    total_latency_ms,
                    latency_window.mean,
                    latency_window.p99,
                    len(detections),
                    active_res,
                    "ON" if seg_enabled else "OFF",
                    "ALERT" if drift_alert else "ok",
                )

            prev_action_idx = held_action.action_index
            prev_detections = detections

    except Exception:
        logger.exception("Unhandled exception in inference loop")
    finally:
        # ═══════════════════════════════════════════════════════════════
        #  GRACEFUL SHUTDOWN
        # ═══════════════════════════════════════════════════════════════
        logger.info("Shutdown sequence started")

        # 0. Release video writer if recording was enabled.
        if video_writer is not None:
            video_writer.release()
            logger.info("Video recording saved to %s", record_path)

        # 1. Stop camera thread and drain queue.
        camera.stop()

        # 2. Flush telemetry buffer to Parquet.
        telemetry_logger.shutdown()

        # 3. Save online-updated RL weights if active.
        if online_enabled and frame_count > 0:
            online_save_dir = agent_dir / "online"
            try:
                controller.agent.save(online_save_dir)
                logger.info("Online RL weights saved to %s", online_save_dir)
            except Exception:
                logger.warning(
                    "Failed to save online RL weights", exc_info=True,
                )

        # 4. Save conformal predictor state (updated alpha + quantile).
        conformal_save = Path("checkpoints/conformal/conformal_state.pt")
        try:
            controller.conformal.save_state(conformal_save)
            logger.info("Conformal state saved to %s", conformal_save)
        except Exception:
            logger.warning(
                "Failed to save conformal state", exc_info=True,
            )

        # 5. Destroy TensorRT engines and free CUDA context.
        if detection_engine is not None:
            detection_engine.shutdown()
        if segmentation_engine is not None:
            segmentation_engine.shutdown()

        # 6. GPU monitor shutdown (NVML cleanup).
        gpu_monitor.shutdown()

        # 7. Terminate dashboard subprocess.
        if dashboard_proc is not None:
            dashboard_proc.terminate()
            try:
                dashboard_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_proc.kill()
            logger.info("Dashboard subprocess terminated")

        logger.info(
            "Shutdown complete — %d frames processed, %d dropped",
            frame_count,
            camera.frames_dropped,
        )
        sys.exit(0)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
