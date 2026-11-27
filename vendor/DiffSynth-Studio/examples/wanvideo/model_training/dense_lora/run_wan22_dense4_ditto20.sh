#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio

BOOTSTRAP=examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py
INFERENCE=examples/wanvideo/model_training/dense_lora/infer_wan22_dense4_resident_batch.py
SELECTION=/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_ditto20_depth_normal/selection_manifest.json
OUTPUT_ROOT=/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_ditto20_depth_normal
LOG_ROOT="$OUTPUT_ROOT/logs"
PYTHON=/pfs/mayuema/dense/wan/envs/cog/bin/python

mkdir -p "$LOG_ROOT"

pids=()
cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for device in 0 1 2 3; do
  (
    set -euo pipefail
    echo "[$(date --iso-8601=seconds)] start resident shard=$device"
    CUDA_VISIBLE_DEVICES=$device "$PYTHON" -u "$BOOTSTRAP" "$INFERENCE" \
      --selection_manifest "$SELECTION" \
      --output_root "$OUTPUT_ROOT" \
      --shard_rank "$device" \
      --shard_count 4 \
      --tasks depth,normal \
      --matte_subject person
    echo "[$(date --iso-8601=seconds)] complete resident shard=$device"
  ) >"$LOG_ROOT/gpu${device}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ $status -ne 0 ]]; then
  echo "At least one Wan2.2 inference worker failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

completed=$(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name INFERENCE_COMPLETE | wc -l)
if [[ $completed -ne 20 ]]; then
  echo "Expected 20 completed cases, found $completed" >&2
  exit 1
fi
"$PYTHON" -c 'import json,sys; json.dump({"cases":20,"tasks":["depth","normal"],"selection_manifest":sys.argv[1]},open(sys.argv[2],"w"),indent=2)' \
  "$SELECTION" "$OUTPUT_ROOT/BATCH_COMPLETE"
echo "[$(date --iso-8601=seconds)] Wan2.2 Ditto20 batch complete"
