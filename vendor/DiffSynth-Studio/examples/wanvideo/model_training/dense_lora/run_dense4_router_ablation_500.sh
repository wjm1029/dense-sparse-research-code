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
DEPTH_CACHE=/pfs/mayuema/dense/wan/cache/depth_latents_bf16_bright40
TARGET_CACHE=/pfs/mayuema/dense/wan/cache/dense4_targets_stability64_v2
PROMPTS=/pfs/mayuema/dense/prompts
MODEL=/pfs/mayuema/dense/wan/Wan2.1-T2V-1.3B

if [[ ! -f "$TARGET_CACHE/CACHE_COMPLETE" ]]; then
  mkdir -p "$TARGET_CACHE"
  "$WAN21_TORCHRUN" --standalone --nproc_per_node=4 \
    examples/wanvideo/model_training/dense_lora/build_dense4_target_cache.py \
    --dataset_root "$DATASET" --depth_cache_dir "$DEPTH_CACHE" \
    --prompt_dir "$PROMPTS" --model_dir "$MODEL" --cache_dir "$TARGET_CACHE" \
    --max_samples 64
fi

for ROUTER_MODE in identity learned; do
  OUTPUT="/pfs/mayuema/dense/wan/outputs/dense4_${ROUTER_MODE}_stability64_500_v2"
  test ! -e "$OUTPUT"
  "$COG_TORCHRUN" --standalone --nproc_per_node=4 "$BOOTSTRAP" \
    examples/wanvideo/model_training/dense_lora/train_dense4_ddp.py \
    --depth_cache_dir "$DEPTH_CACHE" --target_cache_dir "$TARGET_CACHE" \
    --dataset_root "$DATASET" --prompt_dir "$PROMPTS" --model_dir "$MODEL" \
    --output_path "$OUTPUT" --max_train_steps 500 \
    --learning_rate 1e-4 --weight_decay 1e-2 \
    --router_learning_rate 1e-5 --router_weight_decay 1e-2 \
    --router_identity_weight 1e-3 --router_mode "$ROUTER_MODE" \
    --lora_rank 32 --router_hidden_dim 512 \
    --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
    --save_steps 100 --warmup_steps 50 --max_grad_norm 1.0 \
    --num_workers 2 --gradient_checkpointing
done
