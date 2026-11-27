#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 OPENCV_IO_ENABLE_OPENEXR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false
export NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIR=examples/wanvideo/model_training/dense_lora
PY=/pfs/mayuema/dense/wan/envs/cog/bin/python
RUN=/pfs/mayuema/dense/wan/envs/cog/bin/torchrun
OUT=/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_opsd_clean_local12_5250_resume_20261020
STATE=/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_opsd_clean_local12_5250_20261018/trainer_state_latest.pt
if [[ -f "$OUT/TRAINING_COMPLETE" ]]; then exit 0; fi
if [[ -f "$OUT/trainer_state_latest.pt" ]]; then STATE="$OUT/trainer_state_latest.pt"; fi
"$PY" "$DIR/cog_runtime_bootstrap.py" "$DIR/preflight_dense6_opsd_resume_20261020.py" --state "$STATE" --output "$OUT"
"$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_dense6_long477_opsd_ddp.py" \
    --output_dir "$OUT" --max_steps 5250 --resume "$STATE" --skip_initial_eval
test -f "$OUT/TRAINING_COMPLETE"
