#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
DIR=examples/wanvideo/model_training/dense_lora
OUT=/pfs/mayuema/dense/wan/outputs/dense6_flow32_train_fit_120k_dispersion_20261013
CKPT=/pfs/mayuema/dense/wan/outputs/wan22_dense6_flow32_dispersion_100k_to120k_20261012/step-120000.safetensors
indices=(0 1837 3675 5512)
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" /pfs/mayuema/dense/wan/envs/cog/bin/python "$DIR/cog_runtime_bootstrap.py" "$DIR/eval_wan22_dense6_flow32_fit.py" --checkpoint "$CKPT" --clip_indices "${indices[$gpu]}" --output_dir "$OUT/shard_$gpu" > "/pfs/mayuema/dense/wan/logs/dense6_dispersion_eval120k_shard${gpu}_20261013.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" != 0 ]]; then
  echo 'EVAL_FAILED: inspect shard logs'
  exit 1
fi
/pfs/mayuema/dense/wan/envs/cog/bin/python "$DIR/summarize_dense6_dispersion_eval120k_20261013.py"
