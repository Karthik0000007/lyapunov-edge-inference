"""
scripts/build_tensorrt.py
─────────────────────────
Compile ONNX models to TensorRT FP16 engines.  Supports multi-resolution
builds for the detection model (320, 480, 640) and single-resolution build
for the segmentation model (256).

Usage
─────
    # Detection — build 3 resolution engines:
    python scripts/build_tensorrt.py \
        --onnx models/detection/yolov8n.onnx \
        --resolutions 320 480 640 \
        --precision fp16 \
        --output-dir models/detection

    # Segmentation — single resolution:
    python scripts/build_tensorrt.py \
        --onnx models/segmentation/unet.onnx \
        --resolutions 256 \
        --precision fp16 \
        --output-dir models/segmentation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def build_engine(
    onnx_path: Path,
    output_path: Path,
    resolution: int,
    precision: str = "fp16",
    max_workspace_gb: int = 2,
) -> Path:
    """Parse an ONNX model and compile it into a serialised TensorRT engine.

    Parameters
    ----------
    onnx_path:
        Path to the source ONNX file.
    output_path:
        Path for the serialised ``.engine`` file.
    resolution:
        Spatial resolution (H=W) for the optimisation profile.
    precision:
        ``"fp16"`` or ``"fp32"``.
    max_workspace_gb:
        Maximum GPU workspace in GiB for the TensorRT builder.

    Returns
    -------
    Path
        The path to the serialised engine file.
    """
    import tensorrt as trt  # type: ignore[import-untyped]

    trt_logger = trt.Logger(trt.Logger.WARNING)

    # ── Builder + network ────────────────────────────────────────────────
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # ── ONNX parser ──────────────────────────────────────────────────────
    parser = trt.OnnxParser(network, trt_logger)
    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    logger.info(
        "Parsed ONNX model: %d inputs, %d outputs",
        network.num_inputs,
        network.num_outputs,
    )

    # ── Builder config ───────────────────────────────────────────────────
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        max_workspace_gb * (1 << 30),
    )

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            logger.warning("Platform does not have fast FP16 — engine may be slow")
        config.set_flag(trt.BuilderFlag.FP16)

    # ── Optimisation profile (explicit batch) ────────────────────────────
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    # Determine channel count from the ONNX input shape.
    onnx_shape = input_tensor.shape
    channels = onnx_shape[1] if len(onnx_shape) >= 4 else 3

    min_shape = (1, channels, resolution, resolution)
    opt_shape = (1, channels, resolution, resolution)
    max_shape = (1, channels, resolution, resolution)

    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    logger.info(
        "Building engine — resolution=%d  precision=%s  workspace=%d GiB",
        resolution,
        precision,
        max_workspace_gb,
    )

    # ── Build ────────────────────────────────────────────────────────────
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"TensorRT engine build failed for resolution={resolution}")

    # ── Serialise ────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(serialized_engine)
    size_mb = output_path.stat().st_size / (1 << 20)
    logger.info("Engine saved: %s (%.1f MB)", output_path, size_mb)

    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile an ONNX model to a TensorRT FP16 engine.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Path to the source ONNX file.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[320, 480, 640],
        help="Target resolutions (HxW).  Default: 320 480 640.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Compute precision.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/detection"),
        help="Output directory for .engine files.",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=2,
        help="Max builder workspace in GiB.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = parse_args(argv)

    onnx_stem = args.onnx.stem
    for res in args.resolutions:
        engine_name = f"{onnx_stem}_{res}.engine"
        output_path = args.output_dir / engine_name
        build_engine(
            onnx_path=args.onnx,
            output_path=output_path,
            resolution=res,
            precision=args.precision,
            max_workspace_gb=args.workspace,
        )

    logger.info(
        "All engines built — %d file(s) in %s",
        len(args.resolutions),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
