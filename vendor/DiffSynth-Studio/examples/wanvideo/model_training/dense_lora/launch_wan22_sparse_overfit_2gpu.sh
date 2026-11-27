#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=2,3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo
PY=/pfs/mayuema/dense/wan/envs/cog/bin/python
RUN=/pfs/mayuema/dense/wan/envs/cog/bin/torchrun
DIR=examples/wanvideo/model_training/dense_lora
INDEX=/pfs/mayuema/dense/wan/cache/wan22_sparse_coordinates_v1_20261011
OUTPUT=/pfs/mayuema/dense/wan/outputs/wan22_sparse_lora_single_clip_overfit_20261011
test -f "$INDEX/INDEX_COMPLETE"
"$PY" "$DIR/cog_runtime_bootstrap.py" "$DIR/test_wan22_sparse.py"
"$RUN" --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29637 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_sparse_ddp.py" --index_dir "$INDEX" --output_dir "$OUTPUT" --max_steps 1000 --save_steps 100 --eval_steps 50 --eval_samples 1 --overfit_clip auto
