#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

OUTPUT=/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_train_eval4
LOG=/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_train_eval4.log

test -f /pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k/step-10000.safetensors
test ! -e "$OUTPUT/EVAL_COMPLETE"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date --iso-8601=seconds)] Wan2.1 Dense4 train eval4 started"

/opt/conda/envs/cog/bin/torchrun --standalone --nproc_per_node=4 \
  examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py \
  examples/wanvideo/model_training/dense_lora/eval_wan21_dense4_train_cases.py

echo "[$(date --iso-8601=seconds)] Wan2.1 Dense4 train eval4 complete"
