#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export OMP_NUM_THREADS=1

/opt/conda/envs/wan21/bin/torchrun --standalone --nproc_per_node=4 \
  examples/wanvideo/model_training/dense_lora/build_dense4_target_cache.py \
  --dataset_root /pfs/mayuema/dense/dataset/prod_final_9110_20260816/train \
  --depth_cache_dir /pfs/mayuema/dense/wan/cache/depth_latents_bf16_bright40 \
  --prompt_dir /pfs/mayuema/dense/prompts \
  --model_dir /pfs/mayuema/dense/wan/Wan2.1-T2V-1.3B \
  --cache_dir /pfs/mayuema/dense/wan/cache/dense4_targets_bf16_bright40
