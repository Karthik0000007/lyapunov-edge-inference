## Plan: Live Demonstration Recording

TL;DR — Create a 6–10 minute recorded demo showing: (1) a 30s intro & motivation, (2) 60s architecture & assets, (3) a short 2–3 minute environment/setup + commands, (4) a 2–3 minute live run (or pre-recorded fallback), (5) 60–90s results, limitations and next steps. Deliver: `demo_video.mp4`, `demo_transcript.txt`, `demo_README.md`.

---

### Scripted 6–10 Minute Timeline (word-for-word + on-screen cues)

0:00 — 0:30 — Intro (on-camera or voice)
- Spoken script (speak naturally, 14–16s per sentence): "Hello — I'm [Your Name]. This project is 'Lyapunov Edge Inference', a real-time controller that uses learned Lyapunov constraints and visual detection to maintain safe control in embedded settings. Today I'll show how the system runs and the results we achieved."
- On-screen cue: small lower-third with your name and project title for 4s.

0:30 — 1:30 — Motivation & high-level goals
- Spoken script: "Motivation: edge systems require fast perception plus provable stability. Our contribution integrates detection, a learned controller, and a Lyapunov-based safety layer to maintain stability under disturbances. The demonstration focuses on the live pipeline: perception → controller → monitoring."
- On-screen cue: show [ARCHITECTURE.md](ARCHITECTURE.md#L1) + highlight component boxes.

1:30 — 2:30 — Architecture & assets (show files)
- Spoken script: "Key files: `main.py` is the pipeline entrypoint. `demo/record_demo.py` can create annotated demo videos. `app/dashboard.py` shows live metrics. Models and checkpoints live under `models/` and `checkpoints/`. For the demo I'll use `yolov8n.pt` and the `ppo_lyapunov` checkpoint."
- On-screen cue: show file tree (Explorer) and open `main.py` briefly; highlight the CLI flags `--config`, `--source`, `--agent`.

2:30 — 4:30 — Environment setup & exact commands (run live or show terminal)
- Spoken script (narrate each command as you run): "Now I'll prepare the environment and run a short demo. I'll use a Python venv and the pinned requirements from `requirements.txt`. If you have GPU and TensorRT, follow the engine-building steps; if not, the pipeline will run in CPU fallback mode."
- On-screen cue: show terminal; run the commands below (copy/paste each and speak the purpose).

Environment setup (copyable commands):
```bash
# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: build ONNX → TensorRT (GPU machines only)
```bash
# export ONNX (example using provided weight)
python scripts/export_onnx.py --weights yolov8n.pt --output models/yolov8n.onnx
# build TensorRT engine (requires compatible CUDA/TensorRT)
python scripts/build_tensorrt.py --onnx models/yolov8n.onnx --output models/yolov8n.engine
```

Prepare demo input (use an existing video or record a short one)
```bash
# create demo video if you only have frames or want deterministic behavior
python -m demo.record_demo --input demo/demo_input.mp4 --output demo/demo_video.mp4 --checkpoint checkpoints/ppo_lyapunov/
```

Run the pipeline (no dashboard for clean output):
```bash
python main.py --config config/pipeline.yaml --source demo/demo_video.mp4 --agent checkpoints/ppo_lyapunov/ --conformal checkpoints/conformal/ --no-dashboard
```

If you want the Streamlit dashboard for monitoring:
```bash
streamlit run app/dashboard.py
# open http://localhost:8501
```

4:30 — 6:30 — Live run and narration (or play pre-recorded fallback)
- Spoken script: "Now I'll run the pipeline. Watch the detection boxes, the controller action overlay, and the Lyapunov violation monitor. Note how the controller adjusts and keeps the system within safe bounds."
- On-screen cue: terminal + video player side-by-side or Streamlit dashboard. If live run is risky, play `demo_video.mp4` here and narrate events.

6:30 — 7:30 — Results, metrics and limitations
- Spoken script: "Results: the controller maintained stability across disturbances. Key metrics: average reward = [X], violation rate = [Y]. Limitations: the demo uses a pre-recorded environment and requires TensorRT for embedded inference speed. For conference submission, we can include TensorRT engines and a short hardware note."
- On-screen cue: show `results/` figures and a one-slide summary.

7:30 — 8:00 — Conclusion & next steps
- Spoken script: "Thank you — I welcome feedback and questions. I can provide the demo video, the README with environment commands, and the checkpoints used."

---

### OBS / Recording Checklist & Scene Settings (short)
- Resolution: 1920x1080, 30 fps. Bitrate 8–12 Mbps for good quality.
- Mic: external USB mic (condenser) and pop filter. Record audio track separately if possible.
- Scenes: 1) Intro scene (webcam + title), 2) Code + terminal scene (screen share), 3) Demo scene (video player + dashboard), 4) Ending slide.
- Sources per scene: Capture window (IDE), Display capture, Video capture device (webcam), Audio input capture (mic), Media source (local `demo_video.mp4`).
- Hotkeys: Start/Stop recording, switch scenes, mute mic.

On-screen cue cards (short lines to display during recording):
- Card A: "Why this problem matters — 20s"
- Card B: "Run commands (shown in terminal)"
- Card C: "Results & metrics — see slides"

---

### Deliverables to create after recording
- `demo_video.mp4` — final MP4 (H.264 / AAC), 1080p.
- `demo_transcript.txt` — plain transcript (auto + manual corrections).
- `demo_README.md` — exact commands, environment, checkpoints used, and a short guide for reproducing the run.
- `slides_onepager.pdf` — one page with results and next steps.

### Verification
1. Produce `demo/demo_video.mp4` using `demo/record_demo.py` and confirm playback.
2. Run `python main.py --config config/pipeline.yaml --source demo/demo_video.mp4 --agent checkpoints/ppo_lyapunov/` and confirm the pipeline processes frames without fatal errors.
3. Export a short clip (30–60s) and confirm audio alignment and captions.

---

### Quick decisions / options for me to do next
1. Draft full word-for-word spoken script for each timeline line (I can expand the short scripts above into complete sentences to read). 
2. Create `demo_README.md` with the exact commands, expected file locations, and OBS hotkeys. 
3. Generate an OBS JSON scene collection (optional) or a minimal `demo_README.md` file now.

Tell me which option you want next and I will proceed.