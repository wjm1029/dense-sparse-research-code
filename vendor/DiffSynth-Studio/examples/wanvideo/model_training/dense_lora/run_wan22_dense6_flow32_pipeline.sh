#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. OMP_NUM_THREADS=1
cache=/pfs/mayuema/dense/wan/cache/wan22_albedo_delayed_flow32_targets_v1_bf16_bright40
logdir=/pfs/mayuema/dense/wan/logs
mkdir -p "$logdir"
/pfs/mayuema/dense/wan/envs/cog/bin/torchrun --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29538 \
  examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py \
  examples/wanvideo/model_training/dense_lora/build_wan22_flow32_target_cache.py \
  --dataset_root /pfs/mayuema/dense/dataset/prod_final_9110_20260816/train \
  --source_cache_dir /pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40 \
  --reuse_albedo_cache_dir /pfs/mayuema/dense/wan/cache/wan22_albedo_delayed_flow_targets_v1_bf16_bright40 \
  --official_repo /pfs/mayuema/dense/wan/Wan2.2 \
  --model_dir /pfs/mayuema/dense/wan/Wan2.2-TI2V-5B \
  --cache_dir "$cache" 2>&1 | tee "$logdir/wan22_dense6_flow32_cache_20261007.log"
test -f "$cache/CACHE_COMPLETE"
EXTRA_TARGET_CACHE="$cache" EXPECTED_FLOW_SCALE=32 \
OUTPUT_PATH=/pfs/mayuema/dense/wan/outputs/wan22_dense6_hyperlora_bidir_flow32_stage1_50k \
CODE_PATCH=/pfs/mayuema/dense/wan/docs/wan22_dense6_bidir_flow32_runtime_source.patch \
MAX_TRAIN_STEPS=50000 MASTER_PORT=29548 \
bash examples/wanvideo/model_training/dense_lora/train_wan22_dense6_4gpu.sh \
  2>&1 | tee "$logdir/wan22_dense6_flow32_train_20261007.log"
