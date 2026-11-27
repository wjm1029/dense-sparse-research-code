#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export OMP_NUM_THREADS=1

OUTPUT_PATH=${OUTPUT_PATH:-/pfs/mayuema/dense/wan/cache/wan22_albedo_delayed_flow_targets_v1_bf16_bright40}
MAX_SAMPLES=${MAX_SAMPLES:-0}

/pfs/mayuema/dense/wan/envs/cog/bin/torchrun \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT:-29518} \
  examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py \
  examples/wanvideo/model_training/dense_lora/build_wan22_albedo_flow_target_cache.py \
  --dataset_root /pfs/mayuema/dense/dataset/prod_final_9110_20260816/train \
  --source_cache_dir /pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40 \
  --official_repo /pfs/mayuema/dense/wan/Wan2.2 \
  --model_dir /pfs/mayuema/dense/wan/Wan2.2-TI2V-5B \
  --cache_dir "$OUTPUT_PATH" \
  --max_samples "$MAX_SAMPLES"
