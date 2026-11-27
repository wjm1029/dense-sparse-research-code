#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1

PYTHON_ENV=/pfs/mayuema/dense/wan/envs/cog
DATASET_ROOT=/pfs/mayuema/dense/dataset/prod_final_9110_20260816/train
SOURCE_INDEX=/pfs/mayuema/dense/wan/cache/depth_latents_bf16_bright40/dataset_index.json
OFFICIAL_REPO=/pfs/mayuema/dense/wan/Wan2.2
MODEL_DIR=/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B
DEPTH_CACHE=/pfs/mayuema/dense/wan/cache/wan22_depth_latents_v2_bf16_bright40
TARGET_CACHE=/pfs/mayuema/dense/wan/cache/wan22_dense4_targets_v2_bf16_bright40
PROMPT_DIR=/pfs/mayuema/dense/prompts
LOG=/pfs/mayuema/dense/wan/outputs/wan22_full_cache_4gpu.log

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] Wan2.2 full cache started"
"$PYTHON_ENV/bin/python" - <<'PY'
import torch
print(f"python/torch runtime: torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"visible_gpus={torch.cuda.device_count()}")
PY

"$PYTHON_ENV/bin/torchrun" --standalone --nproc_per_node=4 \
  examples/wanvideo/model_training/dense_lora/build_wan22_depth_cache.py \
  --dataset_root "$DATASET_ROOT" \
  --source_dataset_index "$SOURCE_INDEX" \
  --official_repo "$OFFICIAL_REPO" \
  --model_dir "$MODEL_DIR" \
  --cache_dir "$DEPTH_CACHE" \
  --min_rgb_mean 40

test -f "$DEPTH_CACHE/CACHE_COMPLETE"
echo "[$(date --iso-8601=seconds)] Wan2.2 RGB/depth cache complete"

"$PYTHON_ENV/bin/torchrun" --standalone --nproc_per_node=4 \
  examples/wanvideo/model_training/dense_lora/build_wan22_dense4_target_cache.py \
  --dataset_root "$DATASET_ROOT" \
  --depth_cache_dir "$DEPTH_CACHE" \
  --prompt_dir "$PROMPT_DIR" \
  --official_repo "$OFFICIAL_REPO" \
  --model_dir "$MODEL_DIR" \
  --cache_dir "$TARGET_CACHE"

test -f "$TARGET_CACHE/CACHE_COMPLETE"
echo "[$(date --iso-8601=seconds)] Wan2.2 full dense4 cache complete"
