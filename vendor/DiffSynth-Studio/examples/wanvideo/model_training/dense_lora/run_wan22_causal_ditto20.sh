#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio

MODE=${MODE:?Set MODE=causal_zero_shot or causal_trained}
TASKS=${TASKS:-depth,normal,matte,raymap}
SELECTION=/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_ditto20_depth_normal/selection_manifest.json
BOOTSTRAP=examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py
INFERENCE=examples/wanvideo/model_training/dense_lora/infer_wan22_causal_resident_batch.py
PYTHON=/pfs/mayuema/dense/wan/envs/cog/bin/python

case "$MODE" in
  causal_zero_shot)
    CHECKPOINT=/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_lingbot_stage2_20k/step-20000.safetensors
    CODE_PATCH=/pfs/mayuema/dense/wan/docs/wan22_dense4_lingbot_stage2_runtime_source.patch
    ;;
  causal_trained)
    CHECKPOINT=/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_causal_stage1/step-30000.safetensors
    CODE_PATCH=/pfs/mayuema/dense/wan/docs/wan22_causal_training_runtime_source.patch
    ;;
  *)
    echo "Unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT=/pfs/mayuema/dense/wan/outputs/wan22_${MODE}_ditto20
LOG_ROOT="$OUTPUT_ROOT/logs"
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
"$PYTHON" - "$OUTPUT_ROOT/BATCH_COMPLETE" "$MODE" "$TASKS" "$CHECKPOINT" <<'PY'
import json, sys
json.dump({"cases": 20, "mode": sys.argv[2], "tasks": sys.argv[3].split(","), "checkpoint": sys.argv[4]}, open(sys.argv[1], "w"), indent=2)
PY
echo "Wan2.2 $MODE Ditto20 complete: $OUTPUT_ROOT"

