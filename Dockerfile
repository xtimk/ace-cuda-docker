# =============================================================================
# ACE-Step 1.5 — Generic CUDA Dockerfile (RunPod / Vast.ai ready)
# =============================================================================
#
# Builds ACE-Step 1.5 for x86_64 Linux servers with NVIDIA GPUs.
# Uses uv for fast, reproducible dependency installation.
#
# Persistent data (model checkpoints, HuggingFace cache, outputs) is stored
# under /workspace so it survives container restarts on RunPod / Vast.ai.
# Mount your network volume at /workspace and re-use it across pods.
#
# Build (no local source needed — repo is cloned during build):
#   docker build -t acestep .
#
# Run on RunPod / Vast.ai:
#   Set container port 7860 (Gradio) or 8001 (API).
#   Mount your persistent volume at /workspace.
#   The entrypoint will create /workspace/{checkpoints,huggingface,gradio_outputs,output}
#   automatically on first start and symlink them into /app.
#
# Run locally (Gradio UI — default):
#   docker run --gpus all -it --rm \
#     -p 7860:7860 \
#     -v $(pwd)/workspace:/workspace \
#     acestep
#
# Run locally (REST API server):
#   docker run --gpus all -it --rm \
#     -p 8001:8001 \
#     -v $(pwd)/workspace:/workspace \
#     -e ACESTEP_MODE=api \
#     acestep
#
# Override workspace root (default: /workspace):
#   -e WORKSPACE_DIR=/data
#
# =============================================================================

# ==================== Build arguments ====================
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.7

# Optional PyTorch override — use to target GPUs whose CUDA capability is
# below the minimum supported by the version ACE-Step installs by default.
#
# GTX 1070 (sm_61, Pascal) — cu128 dropped sm_61 (min sm_75), try cu121:
#   --build-arg PYTORCH_VERSION=2.7.1
#   --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cu121
#   --build-arg TORCHAUDIO_VERSION=2.7.1
#   --build-arg TORCHVISION_VERSION=0.22.1
#
# torch / torchaudio / torchvision must all be the same release — their
# native libraries reference symbols that only exist in the matching build.
# Leave all empty (default) to keep whatever versions uv sync installs.
ARG PYTORCH_VERSION=""
ARG PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
ARG TORCHAUDIO_VERSION=""
ARG TORCHVISION_VERSION=""

# ==================== Base image ====================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ==================== System packages ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        git \
        curl \
        wget \
        libsndfile1 \
        ffmpeg \
        openssh-server \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# ==================== uv ====================
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ==================== Project source ====================
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /app && rm -rf /app/.git

WORKDIR /app

# ==================== Install dependencies via uv ====================
# Use uv sync with the lockfile for reproducible builds.
# --no-dev skips dev dependencies, --frozen uses exact lockfile versions.
RUN uv sync --frozen --no-dev --python python3.11 && uv cache clean

# ==================== Optional PyTorch override ====================
# Reinstall torch if PYTORCH_VERSION is set (e.g. for older/Pascal GPUs).
ARG PYTORCH_VERSION
ARG PYTORCH_INDEX
# torchaudio and torchvision must match the torch version exactly — their
# native .so files reference symbols that only exist in the matching release.
ARG TORCHAUDIO_VERSION
ARG TORCHVISION_VERSION
RUN if [ -n "${PYTORCH_VERSION}" ]; then \
        AUDIO_PKG=${TORCHAUDIO_VERSION:+"torchaudio==${TORCHAUDIO_VERSION}"} && \
        VISION_PKG=${TORCHVISION_VERSION:+"torchvision==${TORCHVISION_VERSION}"} && \
        echo "Overriding torch==${PYTORCH_VERSION} ${AUDIO_PKG} ${VISION_PKG} from ${PYTORCH_INDEX}" && \
        uv pip install \
            --python /app/.venv/bin/python \
            "torch==${PYTORCH_VERSION}" \
            ${AUDIO_PKG} \
            ${VISION_PKG} \
            --index-url "${PYTORCH_INDEX}" \
            --force-reinstall && \
        uv pip install \
            --python /app/.venv/bin/python \
            "numpy<2" \
            --force-reinstall && \
        uv cache clean; \
    fi

# ==================== Optional extras ====================
# PEFT is required for LoRA training but is not in the lockfile's main group.
# bitsandbytes enables 8-bit AdamW and QLoRA (4-bit base model) — useful on
# 24 GB GPUs to free VRAM during training.
RUN uv pip install --python /app/.venv/bin/python peft bitsandbytes jupyterlab && uv cache clean

# ==================== Runtime directories ====================
RUN mkdir -p /app/checkpoints /app/gradio_outputs /app/output

# ==================== Environment ====================
# Bind to all interfaces for Docker port-mapping
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV ACESTEP_API_HOST=0.0.0.0

# Default startup mode: "gradio" for the web UI, "api" for the REST server
ENV ACESTEP_MODE=gradio

# Auto-initialize models on startup
ENV ACESTEP_INIT_SERVICE=true

# Default models
ENV ACESTEP_CONFIG_PATH=acestep-v15-turbo
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B
ENV ACESTEP_LLM_BACKEND=pt

# Disable tokenizers parallelism warnings
ENV TOKENIZERS_PARALLELISM=false

# Prevent uv run from re-syncing the venv on every invocation.
# Without this, uv detects any manually-overridden package (e.g. the torch
# version override for older GPUs) and reinstalls from the lockfile each start.
ENV UV_NO_SYNC=1

# Persistent volume root (RunPod / Vast.ai mount point)
ENV WORKSPACE_DIR=/workspace

# Point HuggingFace cache into the persistent workspace so model weights
# are downloaded once and reused across pod restarts.
# Can be overridden if WORKSPACE_DIR is changed.
ENV HF_HOME=/workspace/huggingface

# JupyterLab port (set JUPYTER_PORT to override)
ENV JUPYTER_PORT=8888
# Pass JUPYTER_TOKEN at runtime to require token auth; omit for no auth.

# ==================== Ports ====================
# 22 = SSH | 7860 = Gradio web UI | 8001 = REST API | 8888 = JupyterLab
EXPOSE 22 7860 8001 8888

# ==================== Health check ====================
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${GRADIO_PORT:-7860}/ > /dev/null 2>&1 \
     || curl -sf http://localhost:${ACESTEP_API_PORT:-8001}/health > /dev/null 2>&1 \
     || exit 1

# ==================== Entrypoint ====================
COPY <<'ENTRYPOINT_EOF' /app/docker-entrypoint.sh
#!/usr/bin/env bash
set -e

# ---------------------------------------------------------------------------
# SSH setup (vast.ai injects the account public key via PUBLIC_KEY env var)
# ---------------------------------------------------------------------------
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A 2>/dev/null  # generate host keys if missing
mkdir -p /run/sshd
/usr/sbin/sshd              # start SSH daemon in background
echo "SSH       : started (port 22)"

echo "==========================================="
echo "  ACE-Step 1.5"
echo "==========================================="
echo "Mode      : ${ACESTEP_MODE}"
echo "Python    : $(uv run python --version 2>&1)"
echo "PyTorch   : $(uv run python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"

if uv run python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "CUDA      : $(uv run python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU       : $(uv run python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "Memory    : $(uv run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.total_memory/1024**3:.1f} GB")')"
else
    echo "CUDA      : NOT AVAILABLE — running on CPU"
    echo "           (make sure you launched with --gpus all)"
fi
echo "==========================================="

# ---------------------------------------------------------------------------
# Persistent workspace setup (RunPod / Vast.ai)
#
# Create subdirectories inside WORKSPACE_DIR and symlink them into /app so
# the application reads/writes to the persistent volume transparently.
# Works whether or not a real network volume is mounted at WORKSPACE_DIR:
#   - mounted  → data survives pod restarts
#   - unmounted → data lives in the ephemeral container layer (local runs)
# ---------------------------------------------------------------------------
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
echo "Workspace : ${WORKSPACE_DIR}"

for dir in checkpoints gradio_outputs output; do
    mkdir -p "${WORKSPACE_DIR}/${dir}"
    target="/app/${dir}"
    if [ -L "${target}" ]; then
        : # already a symlink — leave it
    elif [ -d "${target}" ]; then
        rmdir "${target}" 2>/dev/null \
            || { echo "WARNING: ${target} is not empty, skipping symlink"; continue; }
        ln -s "${WORKSPACE_DIR}/${dir}" "${target}"
    fi
done

# HuggingFace model cache → workspace so weights are downloaded only once
export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/huggingface}"
mkdir -p "${HF_HOME}"
echo "HF cache  : ${HF_HOME}"
echo "==========================================="

# ---------------------------------------------------------------------------
# JupyterLab — start in background before the main service
# ---------------------------------------------------------------------------
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
uv run jupyter lab \
    --ip=0.0.0.0 \
    --port="${JUPYTER_PORT}" \
    --no-browser \
    --allow-root \
    --NotebookApp.token="${JUPYTER_TOKEN:-}" \
    --NotebookApp.password="" \
    --notebook-dir="${WORKSPACE_DIR}" \
    > /tmp/jupyter.log 2>&1 &
echo "JupyterLab: started on port ${JUPYTER_PORT} (log: /tmp/jupyter.log)"
echo "==========================================="

# Build --init_service flags
INIT_ARGS=""
if [ "${ACESTEP_INIT_SERVICE:-true}" = "true" ]; then
    INIT_ARGS="--init_service true"
    [ -n "${ACESTEP_CONFIG_PATH:-}" ]   && INIT_ARGS="${INIT_ARGS} --config_path ${ACESTEP_CONFIG_PATH}"
    [ -n "${ACESTEP_LM_MODEL_PATH:-}" ] && INIT_ARGS="${INIT_ARGS} --init_llm true --lm_model_path ${ACESTEP_LM_MODEL_PATH}"
    echo "Auto-init    : DiT=${ACESTEP_CONFIG_PATH:-auto}  LM=${ACESTEP_LM_MODEL_PATH:-none}"
fi

if [ "${ACESTEP_MODE}" = "api" ]; then
    echo "Starting REST API server on 0.0.0.0:${ACESTEP_API_PORT:-8001} ..."
    exec uv run python -m acestep.api_server \
        --host "${ACESTEP_API_HOST:-0.0.0.0}" \
        --port "${ACESTEP_API_PORT:-8001}" \
        ${ACESTEP_EXTRA_ARGS:-}
else
    echo "Starting Gradio UI on 0.0.0.0:${GRADIO_PORT:-7860} ..."
    exec uv run python -m acestep.acestep_v15_pipeline \
        --server-name "${GRADIO_SERVER_NAME:-0.0.0.0}" \
        --port "${GRADIO_PORT:-7860}" \
        --backend "${ACESTEP_LLM_BACKEND:-pt}" \
        ${INIT_ARGS} \
        ${ACESTEP_EXTRA_ARGS:-}
fi
ENTRYPOINT_EOF

RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]