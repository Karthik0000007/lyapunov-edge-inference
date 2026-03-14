"""
scripts/train_detection.py
──────────────────────────
Fine-tune YOLOv8-Nano on the NEU-DET steel surface defect dataset (or a
custom dataset) using the Ultralytics API.  Saves the best checkpoint to
``models/detection/``.

Usage
─────
    python scripts/train_detection.py \
        --data dataset/NEU/NEU-DET \
        --epochs 100 \
        --imgsz 640 \
        --batch 16 \
        --device 0
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── NEU-DET VOC-to-YOLO conversion ──────────────────────────────────────────

_NEU_CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def _parse_voc_annotation(xml_path: Path) -> Tuple[str, List[Tuple[int, float, float, float, float]]]:
    """Parse a Pascal-VOC XML annotation into YOLO format.

    Returns
    -------
    (image_filename, list_of (class_id, cx, cy, w, h) normalised to [0, 1])
    """
    tree = ET.parse(xml_path)  # noqa: S314
    root = tree.getroot()

    filename_el = root.find("filename")
    if filename_el is None or filename_el.text is None:
        raise ValueError(f"No <filename> in {xml_path}")
    filename = filename_el.text

    size_el = root.find("size")
    if size_el is None:
        raise ValueError(f"No <size> in {xml_path}")
    img_w = int(size_el.findtext("width", "1"))
    img_h = int(size_el.findtext("height", "1"))

    labels: List[Tuple[int, float, float, float, float]] = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip()
        if name not in _NEU_CLASSES:
            logger.warning("Unknown class '%s' in %s — skipping", name, xml_path)
            continue
        cls_id = _NEU_CLASSES.index(name)

        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = int(bbox.findtext("xmin", "0"))
        ymin = int(bbox.findtext("ymin", "0"))
        xmax = int(bbox.findtext("xmax", "0"))
        ymax = int(bbox.findtext("ymax", "0"))

        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        labels.append((cls_id, cx, cy, w, h))

    return filename, labels


def _prepare_yolo_dataset(neu_root: Path, output_dir: Path) -> Path:
    """Convert NEU-DET VOC layout into Ultralytics YOLO directory structure.

    Creates::

        output_dir/
            dataset.yaml
            images/train/  images/val/
            labels/train/  labels/val/

    Returns the path to ``dataset.yaml``.
    """
    for split_src, split_dst in [("train", "train"), ("validation", "val")]:
        ann_dir = neu_root / split_src / "annotations"
        img_root = neu_root / split_src / "images"

        img_dst = output_dir / "images" / split_dst
        lbl_dst = output_dir / "labels" / split_dst
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if not ann_dir.exists():
            logger.warning("Annotation directory not found: %s", ann_dir)
            continue

        for xml_path in sorted(ann_dir.glob("*.xml")):
            filename, labels = _parse_voc_annotation(xml_path)
            stem = xml_path.stem

            # Locate the corresponding image across class sub-folders.
            img_found = False
            for class_dir in sorted(img_root.iterdir()):
                if not class_dir.is_dir():
                    continue
                candidate = class_dir / filename
                if candidate.exists():
                    dst_img = img_dst / filename
                    if not dst_img.exists():
                        shutil.copy2(candidate, dst_img)
                    img_found = True
                    break

            if not img_found:
                logger.warning("Image not found for annotation %s", xml_path.name)
                continue

            # Write YOLO label file.
            lbl_path = lbl_dst / f"{stem}.txt"
            with lbl_path.open("w", encoding="utf-8") as fh:
                for cls_id, cx, cy, w, h in labels:
                    fh.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # Write dataset.yaml.
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = (
        f"path: {output_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(_NEU_CLASSES)}\n"
        f"names: {_NEU_CLASSES}\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    logger.info("YOLO dataset prepared at %s", output_dir)
    return yaml_path


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    data_path: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    output_dir: Path,
) -> Path:
    """Fine-tune YOLOv8-Nano and return the path to the best checkpoint."""
    from ultralytics import YOLO  # type: ignore[import-untyped]

    # If data_path is a directory (NEU-DET root), convert to YOLO format.
    if data_path.is_dir():
        yolo_dir = output_dir / "yolo_dataset"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        data_yaml = _prepare_yolo_dataset(data_path, yolo_dir)
    else:
        # Assume it's already a dataset.yaml.
        data_yaml = data_path

    logger.info(
        "Starting YOLOv8n training — epochs=%d  imgsz=%d  batch=%d  device=%s",
        epochs, imgsz, batch, device,
    )

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_dir),
        name="yolov8n_finetune",
        exist_ok=True,
        verbose=True,
    )

    # Locate the best checkpoint produced by Ultralytics.
    train_dir = output_dir / "yolov8n_finetune"
    best_pt = train_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = train_dir / "weights" / "last.pt"

    # Copy best checkpoint to canonical models/ location.
    final_dir = Path("models/detection")
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "yolov8n_best.pt"
    shutil.copy2(best_pt, final_path)
    logger.info("Best checkpoint saved to %s", final_path)

    # Log validation mAP.
    metrics_csv = train_dir / "results.csv"
    if metrics_csv.exists():
        import csv

        with metrics_csv.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            if rows:
                last = rows[-1]
                # Ultralytics column names vary slightly; try common keys.
                for key in last:
                    k = key.strip()
                    if "map50" in k.lower() or "map@0.5" in k.lower():
                        logger.info("Final val mAP@0.5: %s", last[key].strip())
                        break

    return final_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8-Nano on a defect detection dataset.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to NEU-DET root directory or a YOLO dataset.yaml file.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: '0' for GPU 0, 'cpu' for CPU.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/detection"),
        help="Directory for training outputs.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = parse_args(argv)
    train(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
