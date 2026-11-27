#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo

export WAN_CAUSAL_FRAMES_PER_CHUNK=${WAN_CAUSAL_FRAMES_PER_CHUNK:-1}
export WAN_CAUSAL_LOCAL_ATTN_FRAMES=${WAN_CAUSAL_LOCAL_ATTN_FRAMES:--1}
export WAN_CAUSAL_SINK_FRAMES=${WAN_CAUSAL_SINK_FRAMES:-0}

OUTPUT_PATH=${OUTPUT_PATH:-/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_causal_stage1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-10000}
MASTER_PORT=${MASTER_PORT:-29519}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_lingbot_stage2_20k/step-20000.safetensors}
INIT_CODE_PATCH=${INIT_CODE_PATCH:-/pfs/mayuema/dense/wan/docs/wan22_dense4_lingbot_stage2_runtime_source.patch}
CODE_PATCH=${CODE_PATCH:-/pfs/mayuema/dense/wan/docs/wan22_causal_training_runtime_source.patch}

test -f "$INIT_CHECKPOINT"
test -f "$INIT_CODE_PATCH"
test "$(git branch --show-current)" = causal-training-wan22
test -z "$(git status --short --untracked-files=no)"
git show --format=email --binary HEAD > "$CODE_PATCH"

/pfs/mayuema/dense/wan/envs/cog/bin/torchrun --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py \
  examples/wanvideo/model_training/dense_lora/train_wan22_causal_ddp.py \
  --depth_cache_dir /pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40 \
  --target_cache_dir /pfs/mayuema/dense/wan/cache/wan22_dense4_targets_bf16_bright40 \
  --lingbot_depth_cache_dir /pfs/mayuema/dense/wan/cache/wan22_lingbot_depth_latents_v2_bf16_bright40 \
  --lingbot_normal_cache_dir /pfs/mayuema/dense/wan/cache/wan22_lingbot_normal_latents_v1_bf16_bright40 \
  --lingbot_mask_cache_dir /pfs/mayuema/dense/wan/cache/wan22_lingbot_validity_v1_bright40 \
  --dataset_root /pfs/mayuema/dense/dataset/prod_final_9110_20260816/train \
  --prompt_dir /pfs/mayuema/dense/prompts \
  --model_dir /pfs/mayuema/dense/wan/Wan2.2-TI2V-5B \
  --output_path "$OUTPUT_PATH" \
  --code_patch_path "$CODE_PATCH" \
  --init_checkpoint "$INIT_CHECKPOINT" \
  --init_code_patch_path "$INIT_CODE_PATCH" \
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
