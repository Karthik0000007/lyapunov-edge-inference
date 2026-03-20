#!/bin/bash
# Record demo with longer input video at 2x slowdown for easier viewing
export CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"

echo "Recording demo from demo_input_long.mp4 with 2x slowdown..."
echo "This will make the video easier to follow."
echo ""

python main.py \
  --config config/pipeline.yaml \
  --source demo/demo_input_long.mp4 \
  --agent checkpoints/ppo_lyapunov/ \
  --record demo/final_demo_slowed.mp4 \
  --slowdown 2.0 \
  --no-dashboard

echo ""
echo "✓ Demo recorded to: demo/final_demo_slowed.mp4"
