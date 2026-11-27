#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo

OUTPUT_PATH=${OUTPUT_PATH:-/pfs/mayuema/dense/wan/outputs/wan22_long477_opsd_prefix_t0_v1}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-25}
MASTER_PORT=${MASTER_PORT:-29541}
ATTENTION_CONTEXT=${ATTENTION_CONTEXT:-full}
INIT_CHECKPOINT=/pfs/mayuema/dense/wan/outputs/wan22_long477_ar_self_forcing_flip_yz_v3/step-3200.safetensors
INIT_CODE_PATCH=/pfs/mayuema/dense/wan/docs/wan22_long477_ar_self_forcing_flip_yz_runtime_source.patch
CODE_PATCH=/pfs/mayuema/dense/wan/docs/wan22_long477_opsd_prefix_t0_runtime_source.patch
ADAPTATION_BASE_COMMIT=5d5ebbd568f67ceda5b28d7a535f99cc29f4e1d2

test "$(git branch --show-current)" = causal-training-wan22
test -z "$(git status --short --untracked-files=no)"
git format-patch --stdout "$ADAPTATION_BASE_COMMIT"..HEAD > "$CODE_PATCH"

/pfs/mayuema/dense/wan/envs/cog/bin/torchrun --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py \
  examples/wanvideo/model_training/dense_lora/train_wan22_long477_opsd_prefix_ddp.py \
  --main_cache_dir /pfs/mayuema/dense/wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40 \
  --main_normal_cache_dir /pfs/mayuema/dense/wan/cache/wan22_long477_normal_flip_yz_v1_bf16_bright40 \
  --lingbot_depth_cache_dir /pfs/mayuema/dense/wan/cache/wan22_lingbot_depth477_latents_v1_bf16_bright40 \
  --lingbot_normal_cache_dir /pfs/mayuema/dense/wan/cache/wan22_lingbot_normal477_latents_v1_bf16_bright40 \
  --model_dir /pfs/mayuema/dense/wan/Wan2.2-TI2V-5B \
  --prompt_dir /pfs/mayuema/dense/prompts \
  --output_path "$OUTPUT_PATH" --code_patch_path "$CODE_PATCH" \
  --init_checkpoint "$INIT_CHECKPOINT" --init_code_patch_path "$INIT_CODE_PATCH" \
  --max_train_steps "$MAX_TRAIN_STEPS" --save_steps "$SAVE_STEPS" \
  --attention_context "$ATTENTION_CONTEXT"
