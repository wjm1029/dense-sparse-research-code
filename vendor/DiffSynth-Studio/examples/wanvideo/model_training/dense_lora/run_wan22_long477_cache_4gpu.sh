#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1

ENV_ROOT=/pfs/mayuema/dense/wan/envs/cog
DATASET_ROOT=/pfs/mayuema/dense/long481_prod3000_20260906/filtered_motion_ok_brightness_gt40/outputs/train
CACHE_ROOT=/pfs/mayuema/dense/wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40
LOG_ROOT=/pfs/mayuema/dense/wan/outputs

mkdir -p "$CACHE_ROOT" "$LOG_ROOT"
exec "$ENV_ROOT/bin/torchrun" --standalone --nproc_per_node=4 \
  examples/wanvideo/model_training/dense_lora/build_wan22_long477_cache.py \
  --dataset_root "$DATASET_ROOT" \
  --official_repo /pfs/mayuema/dense/wan/Wan2.2 \
  --model_dir /pfs/mayuema/dense/wan/Wan2.2-TI2V-5B \
  --cache_dir "$CACHE_ROOT" \
  --min_rgb_mean 40.0 \
  2>&1 | tee -a "$LOG_ROOT/wan22_long477_cache_4gpu.log"
