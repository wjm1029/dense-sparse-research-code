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

OUTPUT_PATH=${OUTPUT_PATH:-/pfs/mayuema/dense/wan/outputs/wan22_dense6_hyperlora_bidir_stage1_50k}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-50000}
SAVE_STEPS=${SAVE_STEPS:-500}
MASTER_PORT=${MASTER_PORT:-29528}
CODE_PATCH=${CODE_PATCH:-/pfs/mayuema/dense/wan/docs/wan22_dense6_bidir_runtime_source.patch}
EXTRA_TARGET_CACHE=${EXTRA_TARGET_CACHE:-/pfs/mayuema/dense/wan/cache/wan22_albedo_delayed_flow_targets_v1_bf16_bright40}
EXPECTED_FLOW_SCALE=${EXPECTED_FLOW_SCALE:-0}

test "$(git branch --show-current)" = causal-training-wan22
test -z "$(git status --short --untracked-files=no)"
git show --format=email --binary HEAD > "$CODE_PATCH"

/pfs/mayuema/dense/wan/envs/cog/bin/torchrun --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py \
  examples/wanvideo/model_training/dense_lora/train_wan22_dense6_ddp.py \
  --depth_cache_dir /pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40 \
  --target_cache_dir /pfs/mayuema/dense/wan/cache/wan22_dense4_targets_bf16_bright40 \
  --extra_target_cache_dir "$EXTRA_TARGET_CACHE" \
  --expected_flow_scale "$EXPECTED_FLOW_SCALE" \
  --dataset_root /pfs/mayuema/dense/dataset/prod_final_9110_20260816/train \
  --prompt_dir /pfs/mayuema/dense/prompts \
  --model_dir /pfs/mayuema/dense/wan/Wan2.2-TI2V-5B \
  --output_path "$OUTPUT_PATH" \
  --code_patch_path "$CODE_PATCH" \
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
  --save_steps "$SAVE_STEPS" \
  --warmup_steps 100 \
  --max_grad_norm 1.0 \
  --num_workers 2 \
  --allow_legacy_v1_cache \
  --gradient_checkpointing
