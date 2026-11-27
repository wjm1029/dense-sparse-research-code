#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 OPENCV_IO_ENABLE_OPENEXR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false
export NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIR=examples/wanvideo/model_training/dense_lora
RUN=/pfs/mayuema/dense/wan/envs/cog/bin/torchrun
EXTRA=/pfs/mayuema/dense/wan/cache/wan22_dense6_long477_extra_flow32_v1_20261017
OUT=/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_self_forcing_dispersion_4800_20261017
LOG=/pfs/mayuema/dense/wan/logs
test -f /pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_smoke6_20261017/SMOKE_COMPLETE
test -f /pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_smoke6_20261017/SMOKE_VERIFIED.json
test -f "$EXTRA/INPUT_VERIFICATION.json"
rg -q 'PASS Dense6 long:' "$LOG/dense6_long477_tests_20261017.log"
if [[ ! -f "$EXTRA/CACHE_COMPLETE" ]]; then
  "$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/build_wan22_dense6_long477_extra.py" \
    --cache_dir "$EXTRA" > "$LOG/dense6_long477_extra_full_20261017.log" 2>&1
fi
test -f "$EXTRA/CACHE_COMPLETE"
if [[ ! -f "$OUT/TRAINING_COMPLETE" ]]; then
  RESUME=()
  if [[ -f "$OUT/trainer_state_latest.pt" ]]; then RESUME=(--resume "$OUT/trainer_state_latest.pt"); fi
  "$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_dense6_long477_ddp.py" \
    --extra_cache "$EXTRA" --output_dir "$OUT" --max_steps 4800 "${RESUME[@]}" \
    > "$LOG/dense6_long477_sf4800_20261017.log" 2>&1
fi
test -f "$OUT/TRAINING_COMPLETE"
