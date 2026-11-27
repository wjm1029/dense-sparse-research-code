#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True

/opt/conda/envs/wan21/bin/python \
  examples/wanvideo/model_training/dense_lora/train_depth_overfit.py \
  --dataset_root /pfs/mayuema/dense/dataset/prod_final_9110_20260816/train \
  --overfit_clip 0a1afa764b9fb300/008f8fd4bbb646de9069 \
  --model_dir /pfs/mayuema/dense/wan/Wan2.1-T2V-1.3B \
  --output_path /pfs/mayuema/dense/wan/outputs/depth_overfit_1clip_200 \
  --max_train_steps 200 \
  --learning_rate 1e-4 \
  --weight_decay 1e-2 \
  --lora_rank 32 \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --save_steps 100 \
  --log_steps 1 \
  --gradient_checkpointing
