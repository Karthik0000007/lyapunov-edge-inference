"""
scripts/export_onnx.py
──────────────────────
Export a trained detection (YOLOv8n) or segmentation (MobileNetV2-UNet)
PyTorch checkpoint to ONNX with dynamic batch axis and ONNX-Simplifier
post-processing.

Usage
─────
    # Detection (YOLOv8n via Ultralytics):
    python scripts/export_onnx.py \
        --model-type detection \
        --checkpoint models/detection/yolov8n_best.pt \
        --output models/detection/yolov8n.onnx \
        --imgsz 640

    # Segmentation (MobileNetV2-UNet):
    python scripts/export_onnx.py \
        --model-type segmentation \
        --checkpoint models/segmentation/mobilenetv2_unet_best.pt \
        --output models/segmentation/unet.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import torch

logger = logging.getLogger(__name__)


# ── Detection export (Ultralytics) ───────────────────────────────────────────

def _export_detection(checkpoint: Path, output: Path, imgsz: int) -> Path:
    """Export YOLOv8n to ONNX via the Ultralytics API."""
    from ultralytics import YOLO  # type: ignore[import-untyped]

    model = YOLO(str(checkpoint))
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=True,
        simplify=True,
        opset=17,
    )
    export_path = Path(export_path)

    # Move to requested output path if different.
    if export_path.resolve() != output.resolve():
        output.parent.mkdir(parents=True, exist_ok=True)
        export_path.replace(output)
        logger.info("Moved ONNX file to %s", output)
    else:
        logger.info("ONNX file saved at %s", output)

    return output


# ── Segmentation export ─────────────────────────────────────────────────────

def _export_segmentation(checkpoint: Path, output: Path) -> Path:
    """Export MobileNetV2-UNet to ONNX with dynamic batch dimension."""
    import onnx  # type: ignore[import-untyped]

    # Lazy import to avoid pulling in the training script at module level.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_segmentation import MobileNetV2UNet

    # Load checkpoint.
    ckpt = torch.load(checkpoint, map_location="cpu")
    num_classes = ckpt.get("num_classes", 1)
    resolution = ckpt.get("resolution", 256)

    model = MobileNetV2UNet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dummy input.
    dummy = torch.randn(1, 3, resolution, resolution)

    # Export.
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info("Raw ONNX exported to %s", output)

    # ONNX-Simplifier pass.
    try:
        import onnxsim  # type: ignore[import-untyped]

        onnx_model = onnx.load(str(output))
        simplified, check = onnxsim.simplify(onnx_model)
        if check:
            onnx.save(simplified, str(output))
            logger.info("ONNX-Simplifier pass succeeded")
        else:
            logger.warning("ONNX-Simplifier check failed — keeping unsimplified model")
    except ImportError:
        logger.warning("onnxsim not installed — skipping simplification")

    # Validate.
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed: %s", output)

    return output


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PyTorch detection or segmentation model to ONNX.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["detection", "segmentation"],
        help="Model type to export.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the ONNX file (.onnx).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for detection export (ignored for segmentation).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = parse_args(argv)

    if args.model_type == "detection":
        _export_detection(args.checkpoint, args.output, args.imgsz)
    else:
        _export_segmentation(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
