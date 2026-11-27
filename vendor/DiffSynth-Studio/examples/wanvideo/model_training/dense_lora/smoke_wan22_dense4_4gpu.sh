#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo

COG_TORCHRUN=/pfs/mayuema/dense/wan/envs/cog/bin/torchrun
BOOTSTRAP=examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py

OUTPUT=${WAN22_SMOKE_OUTPUT:?Set WAN22_SMOKE_OUTPUT}
MAX_STEPS=${WAN22_MAX_STEPS:-4}
RESUME_ARGS=()
if [[ "${WAN22_RESUME:-0}" == "1" ]]; then
  RESUME_ARGS=(--resume_from "$OUTPUT")
fi

"$COG_TORCHRUN" \
  --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=${WAN22_MASTER_PORT:-29508} \
  "$BOOTSTRAP" examples/wanvideo/model_training/dense_lora/train_wan22_dense4_ddp.py \
  --depth_cache_dir /pfs/mayuema/dense/wan/cache/wan22_depth_smoke1_v2_bf16_bright40 \
  --target_cache_dir /pfs/mayuema/dense/wan/cache/wan22_dense4_targets_smoke1_v2_bf16_bright40 \
  --dataset_root /pfs/mayuema/dense/dataset/prod_final_9110_20260816/train \
  --prompt_dir /pfs/mayuema/dense/prompts \
  --model_dir /pfs/mayuema/dense/wan/Wan2.2-TI2V-5B \
  --output_path "$OUTPUT" \
  --code_patch_path /pfs/mayuema/dense/wan/docs/wan22_dense4_runtime_source.patch \
  --max_train_steps "$MAX_STEPS" \
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
  --save_steps 2 \
  --warmup_steps 2 \
  --max_grad_norm 1.0 \
  --num_workers 0 \
  --gradient_checkpointing \
  --distributed_backend nccl \
  "${RESUME_ARGS[@]}"
