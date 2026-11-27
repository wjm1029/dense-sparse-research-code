#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export OMP_NUM_THREADS=1

WAN21_TORCHRUN=/opt/conda/envs/wan21/bin/torchrun
COG_TORCHRUN=/opt/conda/envs/cog/bin/torchrun
BOOTSTRAP=examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py
DATASET=/pfs/mayuema/dense/dataset/prod_final_9110_20260816/train
MODEL=/pfs/mayuema/dense/wan/Wan2.1-T2V-1.3B
CACHE=/pfs/mayuema/dense/wan/cache/depth_smoke4_bright40
OUTPUT=/pfs/mayuema/dense/wan/outputs/depth_smoke4_ddp

test ! -e "$CACHE"
test ! -e "$OUTPUT"
mkdir -p "$CACHE" "$OUTPUT"

"$WAN21_TORCHRUN" --standalone --nproc_per_node=4 \
  examples/wanvideo/model_training/dense_lora/build_depth_cache.py \
  --dataset_root "$DATASET" \
  --model_dir "$MODEL" \
  --cache_dir "$CACHE" \
  --min_rgb_mean 40.0 \
  --max_samples 4

"$COG_TORCHRUN" --standalone --nproc_per_node=4 \
  "$BOOTSTRAP" \
  examples/wanvideo/model_training/dense_lora/train_depth_ddp.py \
  --cache_dir "$CACHE" \
  --model_dir "$MODEL" \
  --output_path "$OUTPUT" \
  --max_train_steps 2 \
  --learning_rate 1e-4 \
  --weight_decay 1e-2 \
  --lora_rank 32 \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --save_steps 1 \
  --num_workers 0 \
  --gradient_checkpointing
