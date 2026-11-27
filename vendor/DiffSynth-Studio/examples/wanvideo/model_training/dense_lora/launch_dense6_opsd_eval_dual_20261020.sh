#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENCV_IO_ENABLE_OPENEXR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DIR=examples/wanvideo/model_training/dense_lora
PY=/pfs/mayuema/dense/wan/envs/cog/bin/python
OUT=/tmp/wan22_dense6_opsd2200_validation_20261020
PIDS=()
for rank in 0 1; do
    RANK="$rank" LOCAL_RANK=0 WORLD_SIZE=2 "$PY" -u "$DIR/cog_runtime_bootstrap.py" \
        "$DIR/eval_dense6_opsd_clean_val_20261020.py" --output_dir "$OUT" > "$OUT/worker_$rank.log" 2>&1 &
    PIDS+=("$!")
done
status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
exit "$status"
