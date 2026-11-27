#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export OMP_NUM_THREADS=1

OUTPUT=/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k
LOG=/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_resume_8000.log

exec > >(tee -a "$LOG") 2>&1
echo "[$(date --iso-8601=seconds)] resume requested from $OUTPUT"

/opt/conda/envs/cog/bin/torchrun --standalone --nproc_per_node=4 \
  examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py \
  examples/wanvideo/model_training/dense_lora/train_dense4_ddp.py \
  --depth_cache_dir /pfs/mayuema/dense/wan/cache/depth_latents_bf16_bright40 \
  --target_cache_dir /pfs/mayuema/dense/wan/cache/dense4_targets_bf16_bright40 \
  --dataset_root /pfs/mayuema/dense/dataset/prod_final_9110_20260816/train \
  --prompt_dir /pfs/mayuema/dense/prompts \
  --model_dir /pfs/mayuema/dense/wan/Wan2.1-T2V-1.3B \
  --output_path "$OUTPUT" \
  --resume_from "$OUTPUT" \
  --max_train_steps 10000 \
  --learning_rate 1e-4 \
  --weight_decay 1e-2 \
  --router_learning_rate 1e-5 \
  --router_weight_decay 1e-2 \
  --router_identity_weight 1e-3 \
  --router_mode learned \
  --lora_rank 32 \
  --router_hidden_dim 512 \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --save_steps 500 \
  --warmup_steps 500 \
  --max_grad_norm 1.0 \
  --num_workers 2 \
  --gradient_checkpointing

echo "[$(date --iso-8601=seconds)] resume run completed"
