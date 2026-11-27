#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 OPENCV_IO_ENABLE_OPENEXR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false
export NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIR=examples/wanvideo/model_training/dense_lora
RUN=/pfs/mayuema/dense/wan/envs/cog/bin/torchrun
ROOT=/pfs/mayuema/dense/wan/outputs
SMOKE="$ROOT/wan22_dense6_long477_opsd_clean_smoke6_20261018"
OUT="$ROOT/wan22_dense6_long477_opsd_clean_local12_5250_20261018"
LOG=/pfs/mayuema/dense/wan/logs
if [[ ! -f "$SMOKE/SMOKE_COMPLETE" ]]; then
  "$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_dense6_long477_opsd_ddp.py" \
    --output_dir "$SMOKE" --max_steps 5250 --run_steps 6 --skip_initial_eval \
    > "$LOG/dense6_long477_opsd_clean_smoke6_20261018.log" 2>&1
fi
test -f "$SMOKE/SMOKE_COMPLETE"
if [[ ! -f "$OUT/TRAINING_COMPLETE" ]]; then
  RESUME=(--resume "$SMOKE/trainer_state_latest.pt")
  if [[ -f "$OUT/trainer_state_latest.pt" ]]; then RESUME=(--resume "$OUT/trainer_state_latest.pt" --skip_initial_eval); fi
  "$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_dense6_long477_opsd_ddp.py" \
    --output_dir "$OUT" --max_steps 5250 "${RESUME[@]}" \
    > "$LOG/dense6_long477_opsd_clean5250_20261018.log" 2>&1
fi
test -f "$OUT/TRAINING_COMPLETE"
