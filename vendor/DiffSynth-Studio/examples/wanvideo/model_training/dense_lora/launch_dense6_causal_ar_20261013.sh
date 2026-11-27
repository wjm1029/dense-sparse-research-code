#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
export WAN_CAUSAL_FRAMES_PER_CHUNK=1 WAN_CAUSAL_LOCAL_ATTN_FRAMES=-1 WAN_CAUSAL_SINK_FRAMES=0 WAN_AR_FRAMES_PER_CHUNK=3
DIR=examples/wanvideo/model_training/dense_lora
OUT=/pfs/mayuema/dense/wan/outputs
DOC=/pfs/mayuema/dense/wan/docs
RUN=/pfs/mayuema/dense/wan/envs/cog/bin/torchrun
"$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/run_wan22_dense6_causal_ar_stage.py" \
  --stage causal --max_steps 45000 \
  --source "$OUT/wan22_dense6_flow32_dispersion_100k_to120k_20261012/step-120000.safetensors" \
  --source_patch "$DOC/wan22_dense6_bidir_flow32_runtime_source.patch" \
  --output "$OUT/wan22_dense6_causal_dispersion_45k_20261013" \
  > /pfs/mayuema/dense/wan/logs/dense6_causal_dispersion45k_20261013.log 2>&1
test -f "$OUT/wan22_dense6_causal_dispersion_45k_20261013/TRAINING_COMPLETE"
"$RUN" --standalone --nproc_per_node=4 "$DIR/cog_runtime_bootstrap.py" "$DIR/run_wan22_dense6_causal_ar_stage.py" \
  --stage ar --max_steps 43500 \
  --source "$OUT/wan22_dense6_causal_dispersion_45k_20261013/step-45000.safetensors" \
  --source_patch "$DOC/wan22_dense6_causal_dispersion_20261013_runtime.json" \
  --output "$OUT/wan22_dense6_ar_tf_dispersion_43500_20261013" \
  > /pfs/mayuema/dense/wan/logs/dense6_ar_tf_dispersion43500_20261013.log 2>&1
test -f "$OUT/wan22_dense6_ar_tf_dispersion_43500_20261013/TRAINING_COMPLETE"
