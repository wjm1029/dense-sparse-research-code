#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1

ENV_ROOT=/pfs/mayuema/dense/wan/envs/cog
OLD_CACHE=/pfs/mayuema/dense/wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40
NEW_CACHE=/pfs/mayuema/dense/wan/cache/wan22_long477_normal_flip_yz_v1_bf16_bright40
LOG=/pfs/mayuema/dense/wan/outputs/wan22_long477_normal_flip_yz_cache_4gpu.log

mkdir -p "$NEW_CACHE" "$(dirname "$LOG")"
exec "$ENV_ROOT/bin/torchrun" --standalone --nproc_per_node=4 \
  examples/wanvideo/model_training/dense_lora/build_wan22_long477_normal_flip_yz_cache.py \
  --dataset_root /pfs/mayuema/dense/long481_prod3000_20260906/filtered_motion_ok_brightness_gt40/outputs/train \
  --source_index "$OLD_CACHE/source_index.json" \
  --official_repo /pfs/mayuema/dense/wan/Wan2.2 \
  --model_dir /pfs/mayuema/dense/wan/Wan2.2-TI2V-5B \
  --cache_dir "$NEW_CACHE" \
  2>&1 | tee -a "$LOG"
