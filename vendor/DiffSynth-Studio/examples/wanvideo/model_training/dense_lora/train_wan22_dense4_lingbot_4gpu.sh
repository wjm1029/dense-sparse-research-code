#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo

OUTPUT_PATH=${OUTPUT_PATH:-/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_lingbot_stage2}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-10000}

/pfs/mayuema/dense/wan/envs/cog/bin/torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29509 \
  examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py \
  examples/wanvideo/model_training/dense_lora/train_wan22_dense4_ddp.py \
  --depth_cache_dir /pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40 \
  --target_cache_dir /pfs/mayuema/dense/wan/cache/wan22_dense4_targets_bf16_bright40 \
  --lingbot_depth_cache_dir /pfs/mayuema/dense/wan/cache/wan22_lingbot_depth_latents_v2_bf16_bright40 \
  --lingbot_normal_cache_dir /pfs/mayuema/dense/wan/cache/wan22_lingbot_normal_latents_v1_bf16_bright40 \
  --lingbot_mask_cache_dir /pfs/mayuema/dense/wan/cache/wan22_lingbot_validity_v1_bright40 \
  --dataset_root /pfs/mayuema/dense/dataset/prod_final_9110_20260816/train \
  --prompt_dir /pfs/mayuema/dense/prompts \
  --model_dir /pfs/mayuema/dense/wan/Wan2.2-TI2V-5B \
  --output_path "$OUTPUT_PATH" \
  --code_patch_path /pfs/mayuema/dense/wan/docs/wan22_dense4_lingbot_stage2_runtime_source.patch \
  --init_checkpoint /pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora/step-10000.safetensors \
  --init_code_patch_path /pfs/mayuema/dense/wan/docs/wan22_dense4_runtime_source.patch \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --learning_rate 1e-4 \
  --weight_decay 1e-2 \
  --router_learning_rate 1e-5 \
  --router_weight_decay 1e-2 \
  --router_identity_weight 1e-3 \
  --router_mode learned \
  --router_freeze_step -1 \
  --lora_rank 32 \
  --router_hidden_dim 512 \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --save_steps 500 \
  --warmup_steps 100 \
  --max_grad_norm 1.0 \
  --num_workers 2 \
  --allow_legacy_v1_cache \
  --gradient_checkpointing
