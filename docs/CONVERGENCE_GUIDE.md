# Reducing Violations from 28% to 1%: Complete Guide

## Problem Analysis

**Current state:** 28% violations after 200 epochs with 30ms budget
**Target:** ~1% violations (constraint_threshold)
**Root cause:** Lambda (λ) learning rate is too slow

## Why 28% Violations Persist

### Lambda Update Formula
```
λ_{k+1} = λ_k + η_λ · (J_C - d_0)
where:
  η_λ = lambda_lr (default 0.01)
  J_C = constraint_cost (0.28 = 28%)
  d_0 = constraint_threshold (0.01 = 1%)
```

### Current Learning Rate
With default `lambda_lr = 0.01`:
- Each epoch: `λ += 0.01 × (0.28 - 0.01) = λ + 0.0027`
- To increase λ by 1.0 requires: **~370 epochs**
- **Conclusion:** 200 epochs is insufficient with default lambda_lr!

---

## ✅ Solution 1: Increase Lambda Learning Rate (RECOMMENDED)

### Quick Fix
```bash
python scripts/train_ppo.py \
    --traces traces/ \
    --latency-budget 30.0 \
    --lambda-lr 0.04 \
    --epochs 300 \
    --output-dir checkpoints/ppo_fast_convergence
```

**Why this works:**
- `lambda_lr = 0.04` (4× faster than default)
- Each epoch: `λ += 0.04 × 0.27 = λ + 0.0108`
- Expected convergence: **~150-200 epochs**

### Expected Progression
```
Epoch   50: ~25% violations  (λ ≈ 0.6)
Epoch  100: ~15% violations  (λ ≈ 1.2)
Epoch  150: ~8% violations   (λ ≈ 1.8)
Epoch  200: ~3% violations   (λ ≈ 2.4)
Epoch  250: ~1% violations   (λ ≈ 3.0) ← TARGET
```

---

## ✅ Solution 2: Continue Training with Current Checkpoint

### Resume from Checkpoint (if λ is already learned)
```bash
# Check current λ value first
python scripts/analyze_training.py --checkpoint checkpoints/ppo_lagrangian

# If λ is already high (>1.0), just continue training
python scripts/train_ppo.py \
    --traces traces/ \
    --latency-budget 30.0 \
    --epochs 400 \
    --output-dir checkpoints/ppo_lagrangian  # Same dir = continue
```

**Note:** Training will resume from saved checkpoint if it exists.

---

## ✅ Solution 3: Two-Phase Training (Conservative)

### Phase 1: Learn with relaxed constraint (faster)
```bash
python scripts/train_ppo.py \
    --traces traces/ \
    --latency-budget 30.0 \
    --constraint-threshold 0.10 \
    --lambda-lr 0.03 \
    --epochs 150 \
    --output-dir checkpoints/ppo_phase1
```
**Target:** Reduce to ~10% violations

### Phase 2: Fine-tune to strict constraint
```bash
python scripts/train_ppo.py \
    --traces traces/ \
    --latency-budget 30.0 \
    --constraint-threshold 0.01 \
    --lambda-lr 0.05 \
    --epochs 150 \
    --output-dir checkpoints/ppo_phase2
```
**Target:** Converge to ~1% violations

---

## ✅ Solution 4: Stricter Budget + Higher Lambda LR

### Aggressive Convergence
```bash
python scripts/train_ppo.py \
    --traces traces/ \
    --latency-budget 25.0 \
    --lambda-lr 0.05 \
    --constraint-threshold 0.01 \
    --epochs 300 \
    --output-dir checkpoints/ppo_aggressive
```

**Trade-off:**
- ✅ Faster convergence (<200 epochs)
- ⚠️ May sacrifice some reward quality
- ✅ Demonstrates stronger constraint enforcement

---

## 📊 Monitoring Convergence

### 1. Check Training Progress
```bash
# Analyze current training
python scripts/analyze_training.py --checkpoint checkpoints/ppo_lagrangian --plot
```

**Look for:**
- **λ increasing:** Should grow from 0.1 → 2.0+
- **Violations decreasing:** Should trend downward
- **Reward stabilizing:** May decrease slightly as constraints tighten

### 2. Watch Real-Time Logs
During training, monitor:
```
PPO Epoch 100/300 — reward=0.0346  viol=15.20%  λ=1.2450  ...
                                    ↑           ↑
                              Decreasing?  Increasing?
```

### 3. Expected Healthy Training
```
✅ λ grows steadily (not stuck at 0.1)
✅ Violations decrease over time (even slowly)
✅ Reward doesn't collapse (should stay positive)
❌ λ stuck near initial value → increase lambda_lr
❌ Violations not decreasing → train longer or increase lambda_lr
❌ Reward → 0 → latency_budget too strict
```

---

## 🎯 Recommended Action Plan

### **Step 1: Quick Win (Run Now)**
```bash
python scripts/train_ppo.py \
    --traces traces/ \
    --latency-budget 30.0 \
    --lambda-lr 0.04 \
    --epochs 300 \
    --output-dir checkpoints/ppo_converged
```

**Expected time:** ~1-2 hours (depending on hardware)
**Expected result:** < 5% violations by epoch 250

---

### **Step 2: Analyze Results**
```bash
python scripts/analyze_training.py \
    --checkpoint checkpoints/ppo_converged \
    --plot
```

**Decision tree:**
- **< 2% violations?** ✅ SUCCESS → Proceed to evaluation
- **5-10% violations?** → Re-run with `--lambda-lr 0.06` and `--epochs 400`
- **> 15% violations?** → Check logs for issues (λ growth, reward collapse)

---

### **Step 3: Evaluate Converged Agent**
```bash
# Test on unseen traces
python scripts/eval_agent.py \
    --checkpoint checkpoints/ppo_converged \
    --test-traces traces/ \
    --budget 30.0

# Stress test under load
python scripts/stress_test.py \
    --checkpoint checkpoints/ppo_converged
```

---

## 🔬 Understanding Lambda Learning Rate

### What λ Does
- **Low λ (0.1-0.5):** Agent prioritizes reward, ignores constraints
- **Medium λ (0.5-2.0):** Agent balances reward and constraints
- **High λ (2.0+):** Agent strictly enforces constraints, may sacrifice reward

### Tuning lambda_lr

| lambda_lr | Convergence Speed | Risk         | Use Case                    |
|-----------|-------------------|--------------|------------------------------|
| 0.01      | Slow (~400 epochs)| Safe         | High quality, patient training |
| 0.03      | Medium (~200 epochs)| Balanced   | **Recommended default**      |
| 0.05      | Fast (~150 epochs)| Aggressive   | Quick iteration, research    |
| 0.10      | Very fast (~100 epochs)| Unstable | Debugging only             |

---

## ❓ FAQ

### Q: Why not just use a stricter threshold (e.g., 0.005 instead of 0.01)?
**A:** The threshold `d_0` is your **target**, not the convergence speed. Changing it to 0.005 means targeting 0.5% violations instead of 1%, but doesn't help you get there faster from 28%. You need higher `lambda_lr` for speed.

### Q: Will higher lambda_lr hurt reward quality?
**A:** Slightly, but it's worth the trade-off:
- Default (0.01): High reward, slow convergence
- Moderate (0.03-0.04): **Best balance** (recommended)
- High (0.05+): Lower reward, but faster convergence

### Q: Can I resume training from my 28% checkpoint?
**A:** Yes! Just use the same `--output-dir` and increase `--lambda-lr` and `--epochs`:
```bash
python scripts/train_ppo.py \
    --traces traces/ \
    --latency-budget 30.0 \
    --lambda-lr 0.05 \
    --epochs 400 \
    --output-dir checkpoints/ppo_lagrangian  # Same = resume
```

### Q: What if violations don't decrease at all?
**A:** Check:
1. Is λ increasing in logs? If not → increase `lambda_lr` drastically (0.05-0.10)
2. Is `latency_budget` too loose? (> P95) → Lower it to 25-30ms
3. Is training converged but high violation? → Increase `lambda_lr` and re-train

---

## 📋 Quick Reference Commands

```bash
# 1. RECOMMENDED: Balanced convergence
python scripts/train_ppo.py --traces traces/ --latency-budget 30.0 --lambda-lr 0.04 --epochs 300

# 2. Fast convergence (lower quality)
python scripts/train_ppo.py --traces traces/ --latency-budget 25.0 --lambda-lr 0.05 --epochs 250

# 3. Resume current checkpoint with higher λ learning
python scripts/train_ppo.py --traces traces/ --latency-budget 30.0 --lambda-lr 0.05 --epochs 400 --output-dir checkpoints/ppo_lagrangian

# 4. Analyze any checkpoint
python scripts/analyze_training.py --checkpoint <checkpoint_dir> --plot
```

---

## ✨ Success Criteria

Your training is successful when:
- ✅ **Final violation rate < 2%** (within 2× of target)
- ✅ **λ has grown significantly** (from 0.1 → 2.0+)
- ✅ **Reward is positive and stable** (not collapsed)
- ✅ **Recent violations have low variance** (< 1% std deviation)

Once achieved, proceed to:
1. Evaluation on test traces
2. Stress testing
3. Ablation studies
4. Real-time deployment validation
