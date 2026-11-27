#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/pfs/mayuema/dense/wan/DiffSynth-Studio}
cd "$REPO_DIR"

BOOTSTRAP=examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py
INFERENCE=examples/wanvideo/model_training/dense_lora/infer_wan22_dense4_resident_batch.py
SELECTION=/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_ditto20_depth_normal/selection_manifest.json
CHECKPOINT=/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_lingbot_stage2_20k/step-20000.safetensors
CODE_PATCH=/pfs/mayuema/dense/wan/docs/wan22_dense4_lingbot_stage2_runtime_source.patch
OUTPUT_ROOT=/pfs/mayuema/dense/wan/outputs/wan22_offline_stage2_20k_ditto20
LOG_ROOT="$OUTPUT_ROOT/logs"
PYTHON=/pfs/mayuema/dense/wan/envs/cog/bin/python
TASKS=depth,normal,matte,raymap

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
    CUDA_VISIBLE_DEVICES=$device "$PYTHON" -u "$BOOTSTRAP" "$INFERENCE" \
      --selection_manifest "$SELECTION" \
      --output_root "$OUTPUT_ROOT" \
      --checkpoint "$CHECKPOINT" \
      --code_patch_path "$CODE_PATCH" \
      --shard_rank "$device" \
      --shard_count 4 \
      --tasks "$TASKS" \
      --matte_subject person
  ) >"$LOG_ROOT/gpu${device}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ $status -ne 0 ]]; then
  tail -n 100 "$LOG_ROOT"/*.log
  exit "$status"
fi

completed=$(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name INFERENCE_COMPLETE | wc -l)
if [[ $completed -ne 20 ]]; then
  echo "Expected 20 completed cases, found $completed" >&2
  exit 1
fi
"$PYTHON" - "$OUTPUT_ROOT/BATCH_COMPLETE" "$TASKS" "$CHECKPOINT" <<'PY'
import json, sys
json.dump(
    {"cases": 20, "mode": "offline", "tasks": sys.argv[2].split(","), "checkpoint": sys.argv[3]},
    open(sys.argv[1], "w"),
    indent=2,
)
PY
echo "Wan2.2 offline Stage2-20k Ditto20 complete: $OUTPUT_ROOT"
