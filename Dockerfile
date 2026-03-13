# Base: NVIDIA TensorRT 8.6 with CUDA 11.8 and cuDNN 8 on Ubuntu 22.04
FROM nvcr.io/nvidia/tensorrt:23.04-py3

# ── System metadata ───────────────────────────────────────────────────────────
LABEL maintainer="lyapunov-edge-inference"
LABEL cuda="11.8"
LABEL tensorrt="8.6"
LABEL python="3.10"

# ── Environment variables ─────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/tensorrt:${LD_LIBRARY_PATH}

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3.10-distutils \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        git \
        curl \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 and pip3 resolve to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && python3 -m pip install --upgrade pip==24.0 setuptools==69.1.0 wheel==0.42.0

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace/lyapunov-edge-inference

# ── Install pinned Python dependencies ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# ── Create necessary runtime directories ──────────────────────────────────────
RUN mkdir -p traces results/logs checkpoints/ppo_lyapunov \
             checkpoints/ppo_lagrangian checkpoints/conformal \
             checkpoints/transition_model models/detection models/segmentation

# ── Expose Streamlit dashboard port ──────────────────────────────────────────
EXPOSE 8501

# ── Health check (verifies Python + torch import) ────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'" \
    || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "main.py"]
CMD ["--config", "config/pipeline.yaml", \
     "--agent", "checkpoints/ppo_lyapunov/", \
     "--conformal", "checkpoints/conformal/"]
