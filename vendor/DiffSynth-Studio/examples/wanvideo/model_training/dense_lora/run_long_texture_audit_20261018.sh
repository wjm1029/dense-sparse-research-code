#!/usr/bin/env bash
set -u
export OMP_NUM_THREADS=1
DIR=/pfs/mayuema/dense/wan/DiffSynth-Studio/examples/wanvideo/model_training/dense_lora
OUT=/pfs/mayuema/dense/wan/outputs/long1079_texture_audit_20261018
PIDS=()
for rank in 0 1 2 3; do
  /pfs/mayuema/dense/envs/blender-4.5.12-linux-x64/blender --disable-autoexec --background --threads 1 \
    --python "$DIR/audit_actor_texture_paths_blender_20261018.py" -- \
    --input "$OUT/INPUT.json" --rank "$rank" --world 4 > "$OUT/blender_rank${rank}.log" 2>&1 &
  PIDS+=("$!")
done
STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
exit "$STATUS"
