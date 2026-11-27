#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIR=examples/wanvideo/model_training/dense_lora
OUT=/pfs/mayuema/dense/wan/outputs
RUN=/pfs/mayuema/dense/wan/envs/cog/bin/torchrun
INDEX=/pfs/mayuema/dense/wan/cache/wan22_sparse_long477_coordinates_v1_20261016
SOURCE="$OUT/wan22_sparse_prompt_ar_tf_21750_20261013/step-21750.safetensors"
SMOKE="$OUT/wan22_sparse_long477_sf_smoke1_20261016"
SF="$OUT/wan22_sparse_long477_self_forcing_2400_20261016"
OPSD="$OUT/wan22_sparse_long477_opsd_local12_2625_20261016"
if [[ ! -f "$SMOKE/SMOKE_COMPLETE" ]]; then
"$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_sparse_long477_ddp.py" \
  --phase self_forcing --index_dir "$INDEX" --output_dir "$SMOKE" --init_checkpoint "$SOURCE" \
  --max_steps 2400 --run_steps 1 --skip_initial_eval --min_free_gib 40 \
  > /pfs/mayuema/dense/wan/logs/sparse_long477_sf_smoke1_20261016.log 2>&1
fi
test -f "$SMOKE/SMOKE_COMPLETE"
if [[ ! -f "$SF/TRAINING_COMPLETE" ]]; then
RESUME="$SMOKE/trainer_state_latest.pt"
if [[ -f "$SF/trainer_state_latest.pt" ]]; then RESUME="$SF/trainer_state_latest.pt"; fi
"$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_sparse_long477_ddp.py" \
  --phase self_forcing --index_dir "$INDEX" --output_dir "$SF" --init_checkpoint "$SOURCE" \
  --max_steps 2400 --resume "$RESUME" --skip_initial_eval --min_free_gib 40 \
  > /pfs/mayuema/dense/wan/logs/sparse_long477_sf2400_20261016.log 2>&1
fi
test -f "$SF/TRAINING_COMPLETE"
if [[ ! -f "$OPSD/TRAINING_COMPLETE" ]]; then
EXTRA=()
if [[ -f "$OPSD/trainer_state_latest.pt" ]]; then EXTRA=(--resume "$OPSD/trainer_state_latest.pt" --skip_initial_eval); fi
"$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_sparse_long477_ddp.py" \
  --phase opsd_local --index_dir "$INDEX" --output_dir "$OPSD" --init_checkpoint "$SF/step-2400.safetensors" \
  --max_steps 2625 --save_steps 100 --seed 17 --min_free_gib 32 "${EXTRA[@]}" \
  > /pfs/mayuema/dense/wan/logs/sparse_long477_opsd2625_20261016.log 2>&1
fi
test -f "$OPSD/TRAINING_COMPLETE"
