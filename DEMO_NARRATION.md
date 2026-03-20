# Live Demo Narration Script
**Duration**: ~3 minutes
**Project**: Lyapunov-Constrained RL Edge Inference System

---

## 🎬 Setup Before Recording

1. Open terminal in project directory
2. Have this script ready on second monitor or printed
3. Optional: Record screen + audio simultaneously
4. Test audio levels first

---

## 📋 Full Narration Script

### [0:00 - 0:30] INTRODUCTION (30 seconds)

**[Show project directory or README on screen]**

> "Hello everyone. Today I'm demonstrating a **Lyapunov-constrained reinforcement learning system** for real-time defect detection on edge hardware.
>
> This project addresses a critical challenge in industrial manufacturing: maintaining **high detection accuracy** while enforcing **strict latency constraints** on resource-limited GPUs.
>
> Specifically, we're targeting a hard P99 latency budget of **50 milliseconds per frame** on an NVIDIA RTX 3050 with only 4 gigabytes of VRAM."

---

### [0:30 - 1:00] STARTING THE DEMO (30 seconds)

**[Switch to terminal window]**

> "Let me start the inference pipeline. I'll run this command which processes a pre-recorded video of industrial steel surfaces."

**[Type and execute]:**
```bash
export CUDA_PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
python main.py --config config/pipeline.yaml --source demo/demo_input_long.mp4 --agent checkpoints/ppo_lyapunov/ --record demo/final_demo_slowed.mp4 --slowdown 2.0 --no-dashboard
```

**[While command is loading, continue speaking]:**

> "This loads three main components: a **YOLOv8-Nano detection engine** optimized with TensorRT FP16, a **MobileNetV2-UNet segmentation engine** that runs conditionally, and the **Lyapunov-PPO reinforcement learning agent** that makes real-time control decisions."

---

### [1:00 - 2:00] MAIN DEMO WALKTHROUGH (60 seconds)

**[Video starts playing - point to different parts of screen]**

> "Now the pipeline is running. Let me highlight what you're seeing:
>
> **[Point to top-right corner]**
> In the **top right**, there's a status overlay showing **'DEFECT'** in red when surface anomalies are detected, and **'GOOD'** in green when the surface is clean.
>
> **[Point to bounding boxes]**
> The **green and orange bounding boxes** show detected defects on the steel surface — these include **crazing, inclusions, and patches**. Each detection shows the defect class and confidence score.
>
> **[Point to top-left HUD]**
> The **HUD in the top left** displays real-time metrics:
> - **Latency**: currently around 8 to 15 milliseconds per frame
> - **Resolution**: you'll notice this changes dynamically between 320, 480, and 640 pixels
> - **Action index**: the controller's current configuration
> - **Segmentation status**: toggles between ON and OFF
> - **Detection count**: number of defects found in the current frame
>
> **[Point to blue overlay if visible]**
> When you see this **blue semi-transparent overlay**, that's the segmentation stage activating to provide pixel-precise defect boundaries. The RL agent enables this **only when high-confidence detections** are present and the latency budget allows it."

---

### [2:00 - 2:45] TECHNICAL ARCHITECTURE (45 seconds)

**[Let video continue playing]**

> "The key innovation here is the **three-layer adaptive controller**:
>
> **Layer 1** is the **Lyapunov-PPO agent** — a lightweight neural network making decisions in under 0.1 milliseconds. It selects from 27 different configurations, trading off between resolution, detection threshold, and segmentation on-or-off. The Lyapunov constraint ensures **safety and stability** — preventing the agent from taking actions that would violate our latency budget.
>
> **Layer 2** is the **conformal predictor**, which provides a **formal statistical guarantee**. It computes a 99% confidence upper bound on the expected latency and can override the RL agent if a violation is predicted. This bound **adapts online** to handle distribution shifts like GPU thermal throttling or novel defect patterns.
>
> **Layer 3** is a **rule-based fallback** controller. If the learned layers ever fail, this deterministic safety net takes over immediately, ensuring the system never crashes in production.
>
> The result: **sub-10 millisecond inference** on most frames, with P99 latency staying well under our 50 millisecond budget, while maintaining over **80% of the baseline detection accuracy**."

---

### [2:45 - 3:00] CONCLUSION (15 seconds)

**[Let video continue or show stopping the command with Ctrl+C]**

> "This demonstrates how **reinforcement learning with formal safety constraints** can enable real-time AI on edge hardware without compromising reliability.
>
> The system is production-ready, with full telemetry logging, drift detection, and online model updates.
>
> Thank you for watching, and I'm happy to answer any questions."

**[Stop recording]**

---

## 🎯 Key Phrases to Emphasize

- **"Lyapunov constraint"** — provides safety guarantees
- **"Hard P99 latency budget"** — 50 milliseconds
- **"Three-layer architecture"** — RL + conformal + fallback
- **"Adaptive controller"** — dynamically adjusts quality/latency tradeoff
- **"Real-time decisions"** — under 0.1 ms controller overhead
- **"80% baseline accuracy"** — while meeting latency constraints

---

## 📊 Optional: Extended Technical Details

If you have extra time or questions, explain:

### **RL State Vector** (11 dimensions):
- Last frame latency, mean latency, P99 latency
- Detection count, mean confidence, defect area ratio
- Current resolution/threshold/segmentation config
- GPU utilization and temperature

### **Action Space** (27 configurations):
- 3 resolutions × 3 thresholds × 3 segmentation modes
- Lyapunov critic masks infeasible actions before selection

### **Training**:
- Trained on 100K frames of collected telemetry traces
- PPO with Lagrangian dual variable for constraint violation cost
- Lyapunov critic ensures per-step safety, not just asymptotic

### **Hardware**:
- NVIDIA RTX 3050 (4 GB VRAM) — consumer-grade edge GPU
- Windows 11 with CUDA 12.1
- TensorRT 8.6 for inference optimization

---

## 🎥 Recording Tips

### **Audio**:
- Clear, moderate pace
- Pause briefly between sections
- Speak confidently (you built this!)

### **Visual**:
- Show terminal output clearly (increase font size if needed)
- Point to or highlight specific screen regions as you mention them
- Let the demo video run for at least 30-45 seconds uninterrupted

### **Timing**:
- Practice once before final recording
- If you go over 3 minutes, that's fine! 3-4 minutes is still great
- Natural pacing > rushing through

### **Energy**:
- You're tired, but this is the final push!
- Take a 5-minute break before recording if needed
- One single good take is all you need

---

## ✅ Final Checklist Before Recording

- [ ] Terminal font size large enough to read
- [ ] Audio input working and clear
- [ ] Screen recording software ready
- [ ] This script opened on second monitor or printed
- [ ] Video plays without stuttering (`demo/demo_input_long.mp4` exists)
- [ ] CUDA_PATH environment variable set correctly
- [ ] Deep breath — you got this! 💪

---

**Good luck! You've built something impressive — now show it off!** 🚀
