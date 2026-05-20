# ACE-Step CUDA Docker

CUDA-enabled Docker image for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (AI music generation), deployable on vast.ai, RunPod, or locally.

Build args let you target any GPU generation by pinning CUDA, Python, and PyTorch versions. The entrypoint starts SSH, JupyterLab, and the ACE-Step service automatically.

## Build

### RTX 2000+ / Ada and later (default, CUDA 12.8)
```bash
docker build -t tizm/ace-cuda:1.2.0-cu12.8.1-py3.11-u22.04 .
```

On **Windows** use `^` for line continuation in `cmd.exe`, or `` ` `` in PowerShell.  
On **macOS (Apple Silicon / M3)** add `--platform linux/amd64` — the build works via QEMU emulation but will be slower:
```bash
docker buildx build --platform linux/amd64 -t tizm/ace-cuda:1.2.0-cu12.8.1-py3.11-u22.04 .
```

### GTX 1070 / Pascal (older GPUs — CUDA 12.1 PyTorch)
cu128 dropped sm_61 (Pascal), so PyTorch must be overridden:
```bash
docker build \
    --build-arg PYTORCH_VERSION=2.7.1 \
    --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 \
    --build-arg TORCHAUDIO_VERSION=2.7.1 \
    --build-arg TORCHVISION_VERSION=0.22.1 \
    -t tizm/ace-cuda:pascal .
```

> torch, torchaudio, and torchvision must always be the same release — their native libraries reference symbols that only exist in the matching build.

## Run locally

```bash
docker run --rm --runtime=nvidia --gpus all \
    -p 127.0.0.1:7860:7860 \
    -p 127.0.0.1:8888:8888 \
    -v /path/to/workspace:/workspace \
    tizm/ace-cuda:1.2.0-cu12.8.1-py3.11-u22.04
```

On startup the container prints the URLs for Gradio and JupyterLab.  
Mount `/workspace` to a local directory to persist downloaded model weights across restarts.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ACESTEP_MODE` | `gradio` | `gradio` for web UI, `api` for REST server |
| `ACESTEP_CONFIG_PATH` | `acestep-v15-turbo` | Model variant (e.g. `acestep-v15-xl-base`) |
| `ACESTEP_LM_MODEL_PATH` | `acestep-5Hz-lm-4B` | LM checkpoint |
| `ACESTEP_LLM_BACKEND` | `pt` | Inference backend |
| `ACESTEP_INIT_SERVICE` | `true` | Auto-load models on startup |
| `WORKSPACE_DIR` | `/workspace` | Persistent volume mount point |
| `HF_HOME` | `/workspace/huggingface` | HuggingFace model cache |
| `GRADIO_PORT` | `7860` | Gradio UI port |
| `ACESTEP_API_PORT` | `8001` | REST API port |
| `JUPYTER_PORT` | `8888` | JupyterLab port |
| `JUPYTER_TOKEN` | *(unset)* | Set to require token auth for JupyterLab |
| `PUBLIC_KEY` | *(unset)* | SSH public key — injected automatically by vast.ai |

## vast.ai / RunPod deployment

1. Select **Docker entrypoint** — SSH, JupyterLab, and ACE-Step all start automatically.
2. Expose ports `22`, `7860`, `8001`, `8888`.
3. Mount a network volume at `/workspace` to persist model weights across restarts.
4. **vast.ai**: your account SSH key is injected automatically via `PUBLIC_KEY`.  
   **RunPod**: SSH is handled at the proxy level, no extra setup needed.

## Ports

| Port | Service |
|---|---|
| 22 | SSH |
| 7860 | Gradio web UI |
| 8001 | REST API |
| 8888 | JupyterLab |

## Docker Hub

[`docker.io/tizm/ace-cuda`](https://hub.docker.com/repository/docker/tizm/ace-cuda/)

| Tag | Notes |
|---|---|
| `1.0.0-cu13.1.2` | CUDA 13 test (likely broken) |
| `1.1.0-cu12.8.1-py3.11-u22.04` | Ada+ build |
| `1.2.0-cu12.8.1-py3.11-u22.04` | Ada+ with SSH, JupyterLab, vast.ai/RunPod support |
