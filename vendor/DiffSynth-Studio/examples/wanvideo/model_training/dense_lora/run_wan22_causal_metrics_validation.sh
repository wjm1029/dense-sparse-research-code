#!/usr/bin/env bash
set -euo pipefail

cd /pfs/mayuema/dense/wan/DiffSynth-Studio

PYTHON=/pfs/mayuema/dense/wan/envs/cog/bin/python
BOOTSTRAP=examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py
VALIDATE=examples/wanvideo/model_training/dense_lora/validate_wan22_causal_modes.py
OLD_CHECKPOINT=/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_lingbot_stage2_20k/step-20000.safetensors
NEW_CHECKPOINT=/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_causal_stage1/step-30000.safetensors
OLD_PATCH=/pfs/mayuema/dense/wan/docs/wan22_dense4_lingbot_stage2_runtime_source.patch
NEW_PATCH=/pfs/mayuema/dense/wan/docs/wan22_causal_training_runtime_source.patch
OUTPUT_ROOT=/pfs/mayuema/dense/wan/outputs/wan22_causal_30k_validation
NUM_CLIPS=${NUM_CLIPS:-16}

mkdir -p "$OUTPUT_ROOT/logs"

modes=(offline causal_zero_shot causal_trained)
checkpoints=("$OLD_CHECKPOINT" "$OLD_CHECKPOINT" "$NEW_CHECKPOINT")
patches=("$OLD_PATCH" "$OLD_PATCH" "$NEW_PATCH")
pids=()
for index in 0 1 2; do
  mode=${modes[$index]}
  (
    CUDA_VISIBLE_DEVICES=$index "$PYTHON" -u "$BOOTSTRAP" "$VALIDATE" \
      --mode "$mode" \
      --checkpoint "${checkpoints[$index]}" \
      --code_patch_path "${patches[$index]}" \
      --num_clips "$NUM_CLIPS" \
      --output "$OUTPUT_ROOT/$mode.json"
  ) >"$OUTPUT_ROOT/logs/$mode.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ $status -ne 0 ]]; then
  tail -n 100 "$OUTPUT_ROOT"/logs/*.log
  exit "$status"
fi

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
results = {name: json.loads((root / f"{name}.json").read_text()) for name in (
    "offline", "causal_zero_shot", "causal_trained"
)}
summary = {
    "schema": "wan22_causal_three_way_summary_v1",
    "num_clips": results["offline"]["num_clips"],
    "clip_indices": results["offline"]["clip_indices"],
    "overall_mean_mse": {name: value["overall_mean_mse"] for name, value in results.items()},
    "task_mean_mse": {
        task: {name: value["task_metrics"][task]["mean_mse"] for name, value in results.items()}
        for task in results["offline"]["task_metrics"]
    },
    "causal_zero_shot_gap_vs_offline": (
        results["causal_zero_shot"]["overall_mean_mse"] - results["offline"]["overall_mean_mse"]
    ),
    "trained_recovery_fraction": (
        (results["causal_zero_shot"]["overall_mean_mse"] - results["causal_trained"]["overall_mean_mse"])
        / max(results["causal_zero_shot"]["overall_mean_mse"] - results["offline"]["overall_mean_mse"], 1e-12)
    ),
    "streaming_equivalence": results["causal_trained"]["streaming_equivalence"],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
