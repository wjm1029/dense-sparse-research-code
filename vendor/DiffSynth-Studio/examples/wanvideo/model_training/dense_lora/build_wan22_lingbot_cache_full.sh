#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1

PYTHON_ENV=/pfs/mayuema/dense/wan/envs/cog
DATASET_ROOT=/pfs/mayuema/dense/lingbot/lingbot-depth_wan832_depth81/clips
SOURCE_INDEX=/pfs/mayuema/dense/lingbot/lingbot-depth_wan832_depth81/dataset_index_bright40.json
OFFICIAL_REPO=/pfs/mayuema/dense/wan/Wan2.2
MODEL_DIR=/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B
DEPTH_CACHE=/pfs/mayuema/dense/wan/cache/wan22_lingbot_depth_latents_v2_bf16_bright40
NORMAL_CACHE=/pfs/mayuema/dense/wan/cache/wan22_lingbot_normal_latents_v1_bf16_bright40
LOG=/pfs/mayuema/dense/wan/outputs/wan22_lingbot_cache_full.log

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] LingBot Wan2.2 depth cache started"
"$PYTHON_ENV/bin/torchrun" --standalone --nproc_per_node=4 \
  examples/wanvideo/model_training/dense_lora/build_wan22_depth_cache.py \
  --dataset_root "$DATASET_ROOT" \
  --source_dataset_index "$SOURCE_INDEX" \
  --official_repo "$OFFICIAL_REPO" \
  --model_dir "$MODEL_DIR" \
  --cache_dir "$DEPTH_CACHE" \
  --min_rgb_mean 40

test -f "$DEPTH_CACHE/CACHE_COMPLETE"
echo "[$(date --iso-8601=seconds)] LingBot Wan2.2 depth cache complete"

"$PYTHON_ENV/bin/torchrun" --standalone --nproc_per_node=4 \
  examples/wanvideo/model_training/dense_lora/build_wan22_lingbot_normal_cache.py \
  --dataset_root "$DATASET_ROOT" \
  --depth_cache_dir "$DEPTH_CACHE" \
  --official_repo "$OFFICIAL_REPO" \
  --model_dir "$MODEL_DIR" \
  --cache_dir "$NORMAL_CACHE"

test -f "$NORMAL_CACHE/CACHE_COMPLETE"
echo "[$(date --iso-8601=seconds)] LingBot Wan2.2 normal cache complete"

"$PYTHON_ENV/bin/python" \
  examples/wanvideo/model_training/dense_lora/validate_wan22_lingbot_cache.py \
  --depth_cache_dir "$DEPTH_CACHE" \
  --normal_cache_dir "$NORMAL_CACHE" \
  --workers 24

test "$("$PYTHON_ENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$NORMAL_CACHE/validation_report.json")" = passed
date --iso-8601=seconds > "$NORMAL_CACHE/PIPELINE_COMPLETE"
echo "[$(date --iso-8601=seconds)] LingBot Wan2.2 full cache pipeline complete"
