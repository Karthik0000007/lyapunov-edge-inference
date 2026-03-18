# TensorRT Engine Build Guide

This document describes how to produce the TensorRT `.engine` files required by the detection and segmentation pipelines.

## Prerequisites

| Component | Version |
|---|---|
| TensorRT | 8.6.x |
| CUDA Toolkit | 11.8 |
| NVIDIA Driver | >= 525.x |
| Python | 3.10+ |
| PyTorch | 2.x |

> Engines are **architecture-specific**. An engine built on an RTX 3050 will not load on an A100 (different SM version). You must rebuild on the target GPU.

## Step 1: Train the models

```bash
# YOLOv8-Nano detection model
python scripts/train_detection.py --data data/mvtec_ad --epochs 100 --imgsz 640

# MobileNetV2-UNet segmentation model
python scripts/train_segmentation.py --data data/mvtec_ad --epochs 50
```

Checkpoints are saved to `models/detection/best.pt` and `models/segmentation/best.pt`.

## Step 2: Export to ONNX

```bash
# Detection (YOLOv8n via Ultralytics export)
python scripts/export_onnx.py \
    --model-type detection \
    --checkpoint models/detection/yolov8n_best.pt \
    --output models/detection/yolov8n.onnx \
    --imgsz 640

# Segmentation (MobileNetV2-UNet)
python scripts/export_onnx.py \
    --model-type segmentation \
    --checkpoint models/segmentation/mobilenetv2_unet_best.pt \
    --output models/segmentation/unet.onnx
```

## Step 3: Build TensorRT engines

```bash
# Detection: build one engine per resolution (320, 480, 640)
python scripts/build_tensorrt.py \
    --onnx models/detection/yolov8n.onnx \
    --resolutions 320 480 640 \
    --precision fp16 \
    --output-dir models/detection

# Segmentation: single resolution (256)
python scripts/build_tensorrt.py \
    --onnx models/segmentation/unet.onnx \
    --resolutions 256 \
    --precision fp16 \
    --output-dir models/segmentation
```

This produces:
```
models/
  detection/
    yolov8n_320.engine
    yolov8n_480.engine
    yolov8n_640.engine
  segmentation/
    unet_256.engine
```

### Alternative: trtexec CLI

If you prefer the NVIDIA `trtexec` CLI over the Python builder:

```bash
# Detection at 640x640
trtexec \
    --onnx=models/detection/yolov8n.onnx \
    --saveEngine=models/detection/yolov8n_640.engine \
    --fp16 \
    --workspace=2048 \
    --shapes=images:1x3x640x640

# Repeat for 320 and 480 by changing --shapes and output filename.
```

## Step 4: Verify

```bash
python -c "
from pathlib import Path
engines = [
    'models/detection/yolov8n_320.engine',
    'models/detection/yolov8n_480.engine',
    'models/detection/yolov8n_640.engine',
    'models/segmentation/unet_256.engine',
]
for e in engines:
    p = Path(e)
    status = f'{p.stat().st_size / 1e6:.1f} MB' if p.exists() else 'MISSING'
    print(f'  {e}: {status}')
"
```

## Shortcut: make

```bash
make build-engines
```

This runs training, ONNX export, and TensorRT build in sequence.

## Docker

When using the provided Dockerfile (based on `nvcr.io/nvidia/tensorrt:23.04-py3`), TensorRT and CUDA are pre-installed. Mount your data directory and run:

```bash
docker build -t lyapunov-edge-inference .
docker run --gpus all -v $(pwd)/data:/workspace/lyapunov-edge-inference/data \
    lyapunov-edge-inference \
    python scripts/build_tensorrt.py \
        --onnx models/detection/yolov8n.onnx \
        --resolutions 320 480 640 \
        --precision fp16 \
        --output-dir models/detection
```

## Runtime behaviour when engines are missing

If `.engine` files are not present at startup, `main.py` catches the `FileNotFoundError` and runs in **stub mode**:

- **Detection stub**: returns an empty detection list for every frame.
- **Segmentation stub**: segmentation stage is skipped entirely.
- All other pipeline components (controller, telemetry, dashboard) continue to operate normally.

This allows development, testing, and CI to run without a GPU or TensorRT installation.
