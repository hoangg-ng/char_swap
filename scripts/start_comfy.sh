#!/usr/bin/env bash
# Launch a ComfyUI backend for the pipeline (run_swap.sh uses ports 8188-8190, one per GPU).
# Usage: GPU=1,2 PORT=8188 bash scripts/start_comfy.sh   (nvidia-smi numbering)
# Needs a GPU with ~48 GB (24 GB is too small even with offloading).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/.comfy_tmp"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # GPU ids == nvidia-smi indices
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
cd "$ROOT/ComfyUI"
PORT="${PORT:-8188}"
# separate db/temp per instance so several servers (one per GPU) can share this ComfyUI checkout
# FAST_ARGS (default): SageAttention (int8 attention, ~5-8% faster generation) + comfy-kitchen Triton kernels (int8
# matmul etc., ~38% faster generation); same output quality in our tests. FAST_ARGS="" = plain PyTorch.
exec python main.py --listen 127.0.0.1 --port "$PORT" \
  --output-directory "$ROOT/outputs/comfy" \
  --temp-directory "$ROOT/.comfy_tmp/$PORT" \
  --database-url "sqlite:///$ROOT/.comfy_tmp/comfyui_$PORT.db" \
  ${FAST_ARGS---use-sage-attention --enable-triton-backend} ${EXTRA_ARGS:-}
