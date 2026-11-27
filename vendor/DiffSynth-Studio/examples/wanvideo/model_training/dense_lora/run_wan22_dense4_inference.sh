#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export TOKENIZERS_PARALLELISM=false

if [[ $# -eq 0 ]]; then
  echo "usage: $0 --input_video VIDEO --output_dir DIR [inference options]" >&2
  exit 2
fi

COG_PYTHON=/pfs/mayuema/dense/wan/envs/cog/bin/python
BOOTSTRAP=examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py
INFERENCE=examples/wanvideo/model_training/dense_lora/infer_wan22_dense4.py

exec "$COG_PYTHON" -u "$BOOTSTRAP" "$INFERENCE" "$@"
