# Data — Download & Preparation Instructions

This directory holds dataset preparation notes.  The actual dataset files
live under `dataset/` in the project root.

---

## Supported Datasets

### 1. NEU-DET (Detection)

Steel surface defect dataset with **6 classes** and Pascal-VOC XML annotations.

| Property | Value |
|---|---|
| Classes | crazing, inclusion, patches, pitted\_surface, rolled-in\_scale, scratches |
| Image size | 200 × 200 grayscale |
| Annotations | Pascal-VOC XML (bounding boxes) |
| Train/Val split | Pre-split into `train/` and `validation/` |

**Download**: <https://github.com/wkentaro/neu-det> (or search for "NEU surface
defect database").

**Expected layout**:

```
dataset/NEU/NEU-DET/
├── train/
│   ├── annotations/          # crazing_1.xml, inclusion_1.xml, …
│   └── images/
│       ├── crazing/          # crazing_1.jpg, …
│       ├── inclusion/
│       ├── patches/
│       ├── pitted_surface/
│       ├── rolled-in_scale/
│       └── scratches/
└── validation/
    ├── annotations/
    └── images/               # same 6 sub-folders
```

The training script `scripts/train_detection.py` converts this VOC layout to
YOLO format automatically.

---

### 2. KolektorSDD2 (Segmentation)

Metal surface defect dataset with **binary pixel-level masks**.

| Property | Value |
|---|---|
| Format | Image + ground-truth mask pairs |
| Naming | `{id}.png` (image), `{id}_GT.png` (mask) |
| Train samples | ~2 332 pairs |
| Test samples | ~1 004 pairs |

**Download**: <https://www.vicos.si/resources/kolektorsdd2/>

**Expected layout**:

```
dataset/KolektorSDD2/
├── train/
│   ├── 10000.png
│   ├── 10000_GT.png
│   ├── 10001.png
│   ├── 10001_GT.png
│   └── …
└── test/
    ├── 20000.png
    ├── 20000_GT.png
    └── …
```

The training script `scripts/train_segmentation.py` loads these pairs directly.

---

### 3. MVTec AD (Optional — Anomaly Detection)

15-category anomaly detection benchmark with pixel-level ground truth.

| Property | Value |
|---|---|
| Categories | bottle, cable, capsule, carpet, grid, hazelnut, leather, metal\_nut, pill, screw, tile, toothbrush, transistor, wood, zipper |
| Train | Normal ("good") samples only |
| Test | Normal + defective samples |
| Ground truth | Binary masks per defect type |

**Download**: <https://www.mvtec.com/company/research/datasets/mvtec-ad>

**Expected layout**:

```
dataset/MVTec/
└── {category}/
    ├── train/good/             # Normal training images
    ├── test/
    │   ├── good/
    │   └── {defect_type}/      # Defective test images
    └── ground_truth/
        └── {defect_type}/      # Binary masks
```

---

## End-to-End Pipeline

```bash
# 1. Train detection model (YOLOv8-Nano on NEU-DET)
python scripts/train_detection.py \
    --data dataset/NEU/NEU-DET \
    --epochs 100 --imgsz 640 --batch 16 --device 0

# 2. Train segmentation model (MobileNetV2-UNet on KolektorSDD2)
python scripts/train_segmentation.py \
    --data dataset/KolektorSDD2 \
    --epochs 80 --lr 1e-3 --batch 16 --device cuda:0

# 3. Export to ONNX
python scripts/export_onnx.py \
    --model-type detection \
    --checkpoint models/detection/yolov8n_best.pt \
    --output models/detection/yolov8n.onnx --imgsz 640

python scripts/export_onnx.py \
    --model-type segmentation \
    --checkpoint models/segmentation/mobilenetv2_unet_best.pt \
    --output models/segmentation/unet.onnx

# 4. Build TensorRT engines
python scripts/build_tensorrt.py \
    --onnx models/detection/yolov8n.onnx \
    --resolutions 320 480 640 \
    --output-dir models/detection

python scripts/build_tensorrt.py \
    --onnx models/segmentation/unet.onnx \
    --resolutions 256 \
    --output-dir models/segmentation
```

After step 4 the engine files expected by the pipeline will be in place:

```
models/detection/yolov8n_320.engine
models/detection/yolov8n_480.engine
models/detection/yolov8n_640.engine
models/segmentation/unet_256.engine
```

> **Note**: Rename `unet_256.engine` → `unet_fp16.engine` if the pipeline
> config (`config/pipeline.yaml`) uses that filename.
