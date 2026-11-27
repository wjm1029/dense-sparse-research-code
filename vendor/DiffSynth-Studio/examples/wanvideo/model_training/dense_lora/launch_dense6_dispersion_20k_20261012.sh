#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo
DIR=examples/wanvideo/model_training/dense_lora
exec /pfs/mayuema/dense/wan/envs/cog/bin/torchrun --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/run_wan22_dense6_dispersion_stage.py" \
  --source /pfs/mayuema/dense/wan/outputs/wan22_dense6_hyperlora_bidir_flow32_stage1_50k \
  --output /pfs/mayuema/dense/wan/outputs/wan22_dense6_flow32_dispersion_100k_to120k_20261012 \
  --target_step 120000 --weight 0.01 --tau 1
