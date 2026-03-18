# Lyapunov-Constrained RL for Latency-Bounded Edge Inference

A production-grade, real-time defect detection pipeline for industrial manufacturing that enforces a **hard P99 latency budget of 50 ms** on constrained GPU hardware (NVIDIA RTX 3050, 4 GB VRAM) while maintaining ≥ 80% of baseline detection accuracy under overload.

The core contribution is a **three-layer adaptive compute controller** combining:
1. **Lyapunov-PPO Agent** — a lightweight RL policy that dynamically selects degradation actions (resolution reduction, segmentation skipping, confidence threshold adjustment) based on real-time system telemetry.
2. **Conformal Prediction Safety Certificate** — a distribution-free statistical layer providing formal P99 latency bounds with online adaptation to distribution shift (ACI).
3. **Rule-Based Fallback** — a deterministic emergency controller that ensures absolute safety when the learned layers fail.

> See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical documentation.

---

## Key Features

- **YOLOv8-Nano + TensorRT FP16** detection engine (8–15 ms) with three pre-compiled resolution engines (320/480/640) for zero-overhead switching
- **MobileNetV2-UNet + TensorRT FP16** conditional segmentation engine (5–12 ms, only on detected ROIs)
- **Lyapunov action masking** — per-step feasibility enforced at every controller decision, not just asymptotically
- **Adaptive Conformal Inference (ACI)** — quantile updated online to handle thermal throttling, lighting changes, and novel defect types
- **Streamlit dashboard** — real-time latency gauges, controller action timeline, conformal bound vs. actual, annotated live feed
- **Parquet telemetry** — columnar per-frame logging with drift detection (KS-test + CUSUM)

---

## Architecture Overview

```
Input Camera (30–60 FPS)
    → Preprocessing (CLAHE, tiling, edge-channel fusion, ROI crop)
    → Stage 1: YOLOv8-Nano TensorRT FP16  (~8–15 ms)
    → Stage 2: MobileNetV2-UNet TensorRT FP16  (~5–12 ms, conditional)
    → Post-processing (NMS, overlay)
    → Telemetry Logger → Streamlit Dashboard

CPU (parallel):
    State Extractor (s ∈ ℝ¹¹)
    → Lyapunov-PPO Actor  (< 0.01 ms)
    → Lyapunov Action Masking  (< 0.10 ms)
    → Conformal Override  (< 0.02 ms)
    → Rule-Based Fallback
    → Config update (resolution / threshold / seg on-off)
```

---

## Requirements

| Component | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.x |
| TensorRT | 8.6.x |
| CUDA Toolkit | 11.8 |
| NVIDIA Driver | ≥ 525.x |
| OpenCV | 4.8+ (with optional CUDA modules) |
| Streamlit | 1.30+ |
| GPU | NVIDIA RTX 3050 (4 GB VRAM) or any GPU with Compute Capability ≥ 7.5 |
| OS | Ubuntu 20.04+ (recommended) or Windows 10/11 with WSL2 |

---

## Installation

**Option 1: Direct**

```bash
git clone <repository-url>
cd lyapunov-edge-inference

python -m venv venv
source venv/bin/activate        # Linux/macOS
# .\venv\Scripts\activate       # Windows

pip install -r requirements.txt

# Verify CUDA and TensorRT
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
python -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"
```

**Option 2: Docker**

```bash
docker build -t lyapunov-edge-inference .
docker run --gpus all -p 8501:8501 -v $(pwd)/data:/app/data lyapunov-edge-inference
```

---

## Quick Start (End-to-End)

```bash
# 1. Download datasets
#    Follow instructions in data/README.md
#    NEU-DET      → detection training  (6-class steel defects, VOC bbox format)
#    KolektorSDD2 → segmentation training (binary pixel masks)
#    MVTec AD subset → generalization/ablation only (metal, PCB, cable; no bbox labels)

# 2. Train and export models
python scripts/train_detection.py    --data dataset/NEU/NEU-DET --epochs 150 --imgsz 640
python scripts/train_segmentation.py --data dataset/KolektorSDD2 --epochs 120

python scripts/export_onnx.py   --model detection    --checkpoint models/detection/best.pt
python scripts/export_onnx.py   --model segmentation --checkpoint models/segmentation/best.pt
python scripts/build_tensorrt.py --onnx models/detection/yolov8n.onnx  --precision fp16 --resolutions 320 480 640
python scripts/build_tensorrt.py --onnx models/segmentation/unet.onnx  --precision fp16

# 3. Collect traces (rule-based + ε-greedy, ~1 h)
python scripts/collect_traces.py --source camera --duration 3600 --epsilon 0.1

# 4. Train RL agent
python scripts/train_ppo.py      --traces traces/ --epochs 10 --constraint-threshold 0.01
python scripts/train_lyapunov.py --traces traces/ --epochs 10 --pretrained checkpoints/ppo_lagrangian/

# 5. Calibrate conformal predictor
python scripts/calibrate_cp.py --traces traces/ --alpha 0.01

# 6. Run the full pipeline
python main.py \
  --config    config/pipeline.yaml \
  --agent     checkpoints/ppo_lyapunov/ \
  --conformal checkpoints/conformal/

# 7. Open the dashboard
streamlit run app/dashboard.py
# → http://localhost:8501
```

---

## Repository Structure

```
lyapunov-edge-inference/
├── ARCHITECTURE.md          # Full technical documentation
├── README.md                # This file
├── requirements.txt         # Pinned Python dependencies
├── Dockerfile
├── config/
│   ├── pipeline.yaml        # Pipeline parameters
│   ├── controller.yaml      # RL hyperparameters
│   └── deployment.yaml      # Production overrides
├── src/
│   ├── camera.py            # Frame acquisition
│   ├── preprocess.py        # CLAHE, tiling, edge fusion, ROI crop
│   ├── detection.py         # YOLOv8-Nano TensorRT wrapper
│   ├── segmentation.py      # MobileNetV2-UNet TensorRT wrapper
│   ├── controller.py        # Three-layer controller orchestrator
│   ├── agent_lyapunov_ppo.py
│   ├── lyapunov.py          # Lyapunov critic + action masking
│   ├── lagrangian.py        # Lagrangian dual variable (baseline)
│   ├── conformal.py         # Conformal prediction + ACI
│   ├── latency_predictor.py
│   ├── baselines.py         # Rule-based, PID, fixed-config controllers
│   ├── env.py               # Gym environment (trace replay)
│   ├── monitoring.py        # Telemetry + Parquet logging
│   └── drift.py             # KS-test + CUSUM drift detection
├── scripts/
│   ├── collect_traces.py
│   ├── train_detection.py / train_segmentation.py
│   ├── export_onnx.py / build_tensorrt.py
│   ├── train_ppo.py / train_lyapunov.py
│   ├── calibrate_cp.py
│   ├── eval_agent.py / eval_baselines.py
│   ├── stress_test.py / ablation.py
│   └── plot_results.py
├── app/
│   └── dashboard.py         # Streamlit dashboard
├── models/                  # TensorRT engines (detection 320/480/640, segmentation)
├── checkpoints/             # RL weights + conformal state
├── traces/                  # Logged telemetry (Parquet)
├── data/                    # NEU-DET + KolektorSDD2 + MVTec AD subset + calibration images
├── results/                 # Figures, tables, experiment logs
├── tests/                   # Unit + integration tests
└── demo/                    # Demo recording script + video
```

---

## TensorRT Engine Build

The pipeline requires pre-compiled TensorRT `.engine` files for detection (320/480/640) and segmentation (256). See **[docs/ENGINE_BUILD.md](docs/ENGINE_BUILD.md)** for full build instructions, or use the shortcut:

```bash
make build-engines
```

### Runtime fallback when engines are missing

If `.engine` files are absent at startup, `main.py` runs in **stub mode**:

- **Detection**: returns an empty detection list per frame (no crash).
- **Segmentation**: segmentation stage is skipped entirely.
- All other components (RL controller, telemetry, conformal predictor, dashboard) operate normally.

This allows CI, tests, and development to run without a GPU or TensorRT installation.

---

## Evaluation

Run the full evaluation suite against all baselines:

```bash
python scripts/eval_baselines.py --traces traces/ --n-frames 10000
python scripts/eval_agent.py     --traces traces/ --agent checkpoints/ppo_lyapunov/ --n-frames 10000
python scripts/stress_test.py    --config config/pipeline.yaml
python scripts/plot_results.py   --results results/
```

**Tracked metrics:** P50/P95/P99 latency, violation rate (target ≤ 1%), detection quality (mAP@0.5 ≥ 80% of baseline), conformal coverage (≥ 99%), controller overhead (< 0.15 ms).

**Baselines:** Fixed-High-Quality, Fixed-Low-Latency, Rule-Based, PID, PPO (unconstrained), PPO + Lagrangian, PPO + Lyapunov, PPO + Lyapunov + CP (full system).

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |
| `PIPELINE_CONFIG` | `config/pipeline.yaml` | Pipeline configuration path |
| `CONTROLLER_CONFIG` | `config/controller.yaml` | Controller hyperparameter path |
| `CAMERA_SOURCE` | `0` | Camera device index or video file path |

---

## References

- Chow, Nachum, Duéñez-Guzmán, Ghavamzadeh. *A Lyapunov-based Approach to Safe Reinforcement Learning.* NeurIPS 2018.
- Gibbs & Candès. *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021.