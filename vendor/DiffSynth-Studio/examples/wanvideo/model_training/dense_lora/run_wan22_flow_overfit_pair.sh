#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export PYTHONPATH=.
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True
experiment=/pfs/mayuema/dense/wan/outputs/flow_scale_experiment_20261007
test -f "$experiment/SCALE_COMPLETE"
for variant in baseline selected; do
  if [ "$variant" = baseline ]; then device=0; else device=1; fi
  tmux new-session -d -s "wan22_flow_overfit_$variant" \
    "cd /pfs/mayuema/dense/wan/DiffSynth-Studio && set -o pipefail; CUDA_VISIBLE_DEVICES=$device PYTHONPATH=. OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True /pfs/mayuema/dense/wan/envs/cog/bin/python -u examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py examples/wanvideo/model_training/dense_lora/experiment_wan22_flow_scale.py --phase overfit --variant $variant --steps 200 --output_dir $experiment 2>&1 | tee /pfs/mayuema/dense/wan/logs/flow_overfit_${variant}_20261007.log"
done
