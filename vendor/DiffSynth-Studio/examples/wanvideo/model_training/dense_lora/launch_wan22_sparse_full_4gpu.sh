#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export NCCL_SHM_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo
RUN=/pfs/mayuema/dense/wan/envs/cog/bin/torchrun
DIR=examples/wanvideo/model_training/dense_lora
INDEX=/pfs/mayuema/dense/wan/cache/wan22_sparse_coordinates_v1_20261011
OUTPUT=/pfs/mayuema/dense/wan/outputs/wan22_sparse_lora_bidir_full_4gpu_overnight_20261012
test -f "$INDEX/INDEX_COMPLETE"
"$RUN" --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29638 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_sparse_full_ddp.py" --index_dir "$INDEX" --output_dir "$OUTPUT" --max_steps 30000 --max_hours 12 --save_steps 500 --eval_steps 500 --eval_samples 372
