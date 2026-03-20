#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  DEMO COMMAND - Copy and paste this into your terminal
# ═══════════════════════════════════════════════════════════════════════

# Step 1: Set correct CUDA path (REQUIRED - fixes PyCUDA issue)
export CUDA_PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"

# Step 2: Run the demo (paste entire command below)
python main.py \
  --config config/pipeline.yaml \
  --source demo/demo_input_long.mp4 \
  --agent checkpoints/ppo_lyapunov/ \
  --record demo/final_demo_slowed.mp4 \
  --slowdown 2.0 \
  --no-dashboard

# ═══════════════════════════════════════════════════════════════════════
#  What this does:
# ═══════════════════════════════════════════════════════════════════════
#  • Loads 100-second industrial steel defect video
#  • Runs detection + segmentation + RL controller
#  • Records output to demo/final_demo_slowed.mp4
#  • Slows down 2x for easier viewing
#  • Shows real-time defect detection with status overlay
# ═══════════════════════════════════════════════════════════════════════
#  To stop: Press Ctrl+C
#  Output will be saved automatically to demo/final_demo_slowed.mp4
# ═══════════════════════════════════════════════════════════════════════
