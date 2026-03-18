#!/bin/bash
# Exploration-optimized training experiments
# Generated from traces: C:\Users\saira\Desktop\New folder (4)\lyapunov-edge-inference\traces

# Experiment 1: conservative_exploration
# Slightly increased exploration with moderate budget
echo 'Starting experiment: conservative_exploration'
python scripts/train_ppo.py \
    --traces traces \
    --epochs 200 \
    --bc-epochs 50 \
    --lr 0.0003 \
    --latency-budget 25.0 \
    --constraint-threshold 0.01 \
    --rollout-length 1000 \
    --batch-size 256 \
    --output-dir config\exploration\checkpoints_conservative_exploration
echo 'Completed: conservative_exploration'

# Experiment 2: aggressive_exploration
# High exploration with strict latency budget
echo 'Starting experiment: aggressive_exploration'
python scripts/train_ppo.py \
    --traces traces \
    --epochs 150 \
    --bc-epochs 50 \
    --lr 0.0003 \
    --latency-budget 20.0 \
    --constraint-threshold 0.01 \
    --rollout-length 1000 \
    --batch-size 256 \
    --output-dir config\exploration\checkpoints_aggressive_exploration
echo 'Completed: aggressive_exploration'

# Experiment 3: progressive_budget
# Budget starts strict and relaxes during training
echo 'Starting experiment: progressive_budget'
python scripts/train_ppo.py \
    --traces traces \
    --epochs 200 \
    --bc-epochs 50 \
    --lr 0.0003 \
    --latency-budget 22.0 \
    --constraint-threshold 0.01 \
    --rollout-length 1000 \
    --batch-size 256 \
    --output-dir config\exploration\checkpoints_progressive_budget
echo 'Completed: progressive_budget'

# Experiment 4: curiosity_driven
# Encourage diverse state visitation
echo 'Starting experiment: curiosity_driven'
python scripts/train_ppo.py \
    --traces traces \
    --epochs 200 \
    --bc-epochs 50 \
    --lr 0.0003 \
    --latency-budget 28.0 \
    --constraint-threshold 0.01 \
    --rollout-length 1500 \
    --batch-size 256 \
    --output-dir config\exploration\checkpoints_curiosity_driven
echo 'Completed: curiosity_driven'

