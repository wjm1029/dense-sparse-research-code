#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export OMP_NUM_THREADS=1

TORCHRUN=/pfs/mayuema/dense/wan/envs/cog/bin/torchrun
BOOTSTRAP=examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py
TRAIN=examples/wanvideo/model_training/dense_lora/train_wan22_dense4_ddp.py
VALIDATE=examples/wanvideo/model_training/dense_lora/validate_wan22_dense4_checkpoint.py
DATASET=/pfs/mayuema/dense/dataset/prod_final_9110_20260816/train
DEPTH_CACHE=/pfs/mayuema/dense/wan/cache/wan22_depth_smoke1_v2_bf16_bright40
TARGET_CACHE=/pfs/mayuema/dense/wan/cache/wan22_dense4_targets_smoke1_v2_bf16_bright40
PROMPTS=/pfs/mayuema/dense/prompts
MODEL=/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B
CODE_PATCH=/pfs/mayuema/dense/wan/docs/wan22_dense4_runtime_source.patch
OUTPUT=${WAN22_SMOKE_OUTPUT:-/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_smoke1}

test ! -e "$OUTPUT"

"$TORCHRUN" --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29507 "$BOOTSTRAP" "$TRAIN" \
  --depth_cache_dir "$DEPTH_CACHE" \
  --target_cache_dir "$TARGET_CACHE" \
  --dataset_root "$DATASET" \
  --prompt_dir "$PROMPTS" \
  --model_dir "$MODEL" \
  --output_path "$OUTPUT" \
  --code_patch_path "$CODE_PATCH" \
  --max_train_steps 4 \
  --learning_rate 1e-4 \
  --weight_decay 1e-2 \
  --router_learning_rate 1e-5 \
  --router_weight_decay 1e-2 \
  --router_identity_weight 1e-3 \
  --router_mode learned \
  --lora_rank 32 \
  --router_hidden_dim 512 \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --save_steps 2 \
  --warmup_steps 2 \
  --max_grad_norm 1.0 \
  --num_workers 0 \
  --gradient_checkpointing \
  --distributed_backend gloo \
  --single_gpu_smoke

/pfs/mayuema/dense/wan/envs/cog/bin/python "$BOOTSTRAP" "$VALIDATE" \
  --checkpoint "$OUTPUT/step-4.safetensors" \
  --depth_cache_dir "$DEPTH_CACHE" \
  --target_cache_dir "$TARGET_CACHE" \
  --model_dir "$MODEL" \
  --prompt_dir "$PROMPTS" \
  --code_patch_path "$CODE_PATCH" \
  --output_json "$OUTPUT/roundtrip_validation.json"
