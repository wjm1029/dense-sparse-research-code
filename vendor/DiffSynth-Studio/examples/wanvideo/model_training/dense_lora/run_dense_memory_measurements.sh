#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

PYTHON=/opt/conda/envs/cog/bin/python
BOOTSTRAP=examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py
SCRIPT=examples/wanvideo/model_training/dense_lora/measure_dense_memory.py
OUTPUT=/pfs/mayuema/dense/wan/outputs/dense_memory_measurements_20260830
mkdir -p "$OUTPUT"

names=(wan21_full_train wan21_inference wan22_full_train wan22_inference)
versions=(wan21 wan21 wan22 wan22)
modes=(full_train inference full_train inference)
pids=()

for device in 0 1 2 3; do
  name=${names[$device]}
  CUDA_VISIBLE_DEVICES=$device "$PYTHON" -u "$BOOTSTRAP" "$SCRIPT" \
    --version "${versions[$device]}" \
    --mode "${modes[$device]}" \
    --output "$OUTPUT/$name.json" \
    >"$OUTPUT/$name.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
for name in "${names[@]}"; do
  test -f "$OUTPUT/$name.json" || status=1
done
exit "$status"
