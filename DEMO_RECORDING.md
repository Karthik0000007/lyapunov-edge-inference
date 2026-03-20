# Demo Recording Guide

## Quick Start

Simply run the recording script:

### Windows
```cmd
record_demo.bat
```

### Linux/Mac/Git Bash
```bash
./record_demo.sh
```

## What It Does

1. **Source**: Uses `demo/demo_input_long.mp4` (100 seconds, longer video)
2. **Slowdown**: 2x slower output (half speed) for easier viewing
3. **Output**: `demo/final_demo_slowed.mp4`
4. **Features shown**:
   - Real-time defect detection with bounding boxes
   - Status overlay (DEFECT/GOOD) in top-right
   - Adaptive resolution switching (320/480/640)
   - Segmentation ON/OFF based on detections
   - Latency tracking and HUD info

## Manual Command

If you want to customize:

```bash
# Fix CUDA path first
export CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

# Run with custom settings
python main.py \
  --config config/pipeline.yaml \
  --source demo/demo_input_long.mp4 \
  --agent checkpoints/ppo_lyapunov/ \
  --record demo/my_demo.mp4 \
  --slowdown 2.0 \
  --no-dashboard
```

## Slowdown Options

- `--slowdown 1.0` = Normal speed (default)
- `--slowdown 2.0` = Half speed (recommended for demos)
- `--slowdown 3.0` = Third speed (very slow, good for technical presentations)

**How it works**: Reduces output FPS. With input at 30 FPS and slowdown 2.0, output is 15 FPS, making the video appear slower and easier to follow.

## Expected Output

- **Video length**: ~100 seconds of processed video
- **File location**: `demo/final_demo_slowed.mp4`
- **Shows**: Detection working on actual steel/industrial surfaces
- **Status overlay**: Will toggle between GOOD (green) and DEFECT (red)

## Troubleshooting

### If detection doesn't work:
```bash
# Verify CUDA path
echo $CUDA_PATH
# Should be: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1 (without \bin)

# Test TensorRT
python -c "import tensorrt; print('TensorRT OK')"

# Test PyCUDA with correct path
export CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
python -c "import pycuda.driver as cuda; print('PyCUDA OK')"
```

### If video is too fast/slow:
Adjust `--slowdown` parameter (higher = slower).

### If you want live camera instead:
```bash
export CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
python main.py --config config/pipeline.yaml --source 0 --agent checkpoints/ppo_lyapunov/ --record demo/camera_demo.mp4 --slowdown 1.0 --no-dashboard
```

## Key Features Visible in Demo

✅ **Real-time Detection**: YOLOv8-Nano with TensorRT FP16
✅ **Adaptive Controller**: RL agent switching resolution/segmentation
✅ **Status Overlay**: Binary classification (DEFECT vs GOOD)
✅ **Latency Tracking**: Sub-50ms inference maintained
✅ **Multi-stage Pipeline**: Detection + conditional segmentation

---

**Happy demoing!** 🎥
