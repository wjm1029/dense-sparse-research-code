#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
DIR=examples/wanvideo/model_training/dense_lora
OUT=/pfs/mayuema/dense/wan/outputs
RUN=/pfs/mayuema/dense/wan/envs/cog/bin/torchrun
COMMON=(--index_dir /pfs/mayuema/dense/wan/cache/wan22_sparse_coordinates_v1_20261011 --save_steps 500 --eval_steps 1000 --eval_samples 372 --max_hours 0)
"$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_sparse_causal_ar_ddp.py" \
  "${COMMON[@]}" --stage causal --max_steps 22500 \
  --init_checkpoint "$OUT/wan22_sparse_prompt_hyperlora_full_4gpu_25k_20261012/step-25000.safetensors" \
  --output_dir "$OUT/wan22_sparse_prompt_causal_22500_20261013" \
  --stop_file /tmp/wan22_sparse_causal_STOP_REQUESTED_20261013 \
  > /pfs/mayuema/dense/wan/logs/sparse_causal22500_20261013.log 2>&1
test -f "$OUT/wan22_sparse_prompt_causal_22500_20261013/TRAINING_COMPLETE"
"$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_sparse_causal_ar_ddp.py" \
  "${COMMON[@]}" --stage ar --max_steps 21750 \
  --init_checkpoint "$OUT/wan22_sparse_prompt_causal_22500_20261013/step-22500.safetensors" \
  --output_dir "$OUT/wan22_sparse_prompt_ar_tf_21750_20261013" \
  --stop_file /tmp/wan22_sparse_ar_STOP_REQUESTED_20261013 \
  > /pfs/mayuema/dense/wan/logs/sparse_ar_tf21750_20261013.log 2>&1
test -f "$OUT/wan22_sparse_prompt_ar_tf_21750_20261013/TRAINING_COMPLETE"
