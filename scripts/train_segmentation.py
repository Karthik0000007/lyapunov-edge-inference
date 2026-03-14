"""
scripts/train_segmentation.py
─────────────────────────────
Train a MobileNetV2-UNet binary segmentation model on the KolektorSDD2
dataset (image + ground-truth mask pairs).  Saves the best checkpoint to
``models/segmentation/``.

Usage
─────
    python scripts/train_segmentation.py \
        --data dataset/KolektorSDD2 \
        --epochs 80 \
        --lr 1e-3 \
        --batch 16 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_SEG_RESOLUTION: int = 256
_IN_CHANNELS: int = 3


# ── Dataset ──────────────────────────────────────────────────────────────────

class DefectSegmentationDataset(Dataset):
    """Loads image–mask pairs from KolektorSDD2 layout.

    Expected directory structure::

        root/
            {id}.png        ← input image
            {id}_GT.png     ← binary ground-truth mask
    """

    def __init__(
        self,
        root: Path,
        resolution: int = _SEG_RESOLUTION,
    ) -> None:
        self.resolution = resolution
        self.pairs: List[Tuple[Path, Path]] = []

        root = Path(root)
        for img_path in sorted(root.glob("*.png")):
            if img_path.stem.endswith("_GT"):
                continue
            gt_path = img_path.parent / f"{img_path.stem}_GT.png"
            if gt_path.exists():
                self.pairs.append((img_path, gt_path))

        if not self.pairs:
            raise FileNotFoundError(
                f"No image/mask pairs found in {root}.  "
                "Expected files like 10000.png + 10000_GT.png."
            )
        logger.info("Loaded %d image-mask pairs from %s", len(self.pairs), root)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, gt_path = self.pairs[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.resolution, self.resolution))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))  # CHW

        mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.resolution, self.resolution),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # 1HW

        return img, mask


# ── MobileNetV2-UNet Model ──────────────────────────────────────────────────

class _ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU6 block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    """Upsample → concat skip → two ConvBnRelu blocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = _ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = _ConvBnRelu(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle spatial size mismatch from encoder.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MobileNetV2UNet(nn.Module):
    """MobileNetV2 encoder with a 3-level U-Net decoder.

    Encoder tap points (from ``torchvision.models.mobilenet_v2``):
        - Skip 1: features[3]  → 24 channels,  H/4
        - Skip 2: features[6]  → 32 channels,  H/8
        - Skip 3: features[13] → 96 channels,  H/16
        - Bottleneck: features[17] → 320 channels, H/32
            (or features[18] including final conv1x1 → 1280 channels)

    Decoder upsamples 3 times back to H/4, then a final upsample ×4 to H.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        features = backbone.features

        # Encoder stages (frozen-option: caller can freeze if desired).
        self.enc1 = nn.Sequential(*features[:4])     # → 24ch, H/4
        self.enc2 = nn.Sequential(*features[4:7])    # → 32ch, H/8
        self.enc3 = nn.Sequential(*features[7:14])   # → 96ch, H/16
        self.enc4 = nn.Sequential(*features[14:18])  # → 320ch, H/32

        # Decoder (3 levels).
        self.dec3 = _DecoderBlock(320, 96, 128)   # H/32 → H/16
        self.dec2 = _DecoderBlock(128, 32, 64)    # H/16 → H/8
        self.dec1 = _DecoderBlock(64, 24, 32)     # H/8  → H/4

        # Final upsample ×4 and 1x1 conv to output.
        self.final_up = nn.Upsample(scale_factor=4, mode="bilinear",
                                    align_corners=False)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)   # 24ch, H/4
        s2 = self.enc2(s1)  # 32ch, H/8
        s3 = self.enc3(s2)  # 96ch, H/16
        s4 = self.enc4(s3)  # 320ch, H/32

        d3 = self.dec3(s4, s3)  # 128ch, H/16
        d2 = self.dec2(d3, s2)  # 64ch, H/8
        d1 = self.dec1(d2, s1)  # 32ch, H/4

        out = self.final_up(d1)  # 32ch, H
        out = self.head(out)     # num_classes, H
        return out


# ── Loss ─────────────────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    """Combined binary cross-entropy + Dice loss."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


# ── Metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_iou(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean IoU for binary segmentation."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - intersection
    if union < 1e-6:
        return 1.0  # Both empty → perfect match.
    return intersection / union


# ── Training loop ────────────────────────────────────────────────────────────

def train(
    data_path: Path,
    epochs: int,
    lr: float,
    batch: int,
    device_str: str,
    output_dir: Path,
) -> Path:
    """Train MobileNetV2-UNet and return the path to the best checkpoint."""
    device = torch.device(device_str)

    # Datasets.
    train_root = data_path / "train"
    test_root = data_path / "test"

    train_ds = DefectSegmentationDataset(train_root, resolution=_SEG_RESOLUTION)
    val_ds = DefectSegmentationDataset(test_root, resolution=_SEG_RESOLUTION)

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # Model.
    model = MobileNetV2UNet(num_classes=1, pretrained=True).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_iou: float = 0.0
    ckpt_dir = Path("models/segmentation")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "mobilenetv2_unet_best.pt"

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_iou_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_iou_sum += compute_iou(logits, masks) * imgs.size(0)
                val_count += imgs.size(0)

        val_iou = val_iou_sum / max(val_count, 1)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %3d/%d  loss=%.4f  val_IoU=%.4f  lr=%.2e",
            epoch, epochs, train_loss, val_iou, current_lr,
        )

        # Save best.
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_iou": val_iou,
                "num_classes": 1,
                "resolution": _SEG_RESOLUTION,
            }, best_path)
            logger.info(
                "  ↳ New best IoU=%.4f — checkpoint saved to %s",
                best_iou, best_path,
            )

    logger.info("Training complete — best val IoU=%.4f", best_iou)
    return best_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2-UNet on defect segmentation dataset.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to KolektorSDD2 root (must contain train/ and test/ sub-dirs).",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/segmentation"),
        help="Directory for training logs.",
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
        lr=args.lr,
        batch=args.batch,
        device_str=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
