#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export NCCL_SHM_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo
DIR=examples/wanvideo/model_training/dense_lora
test ! -e /tmp/wan22_sparse_prompt_full25k_STOP_REQUESTED_20261012
/pfs/mayuema/dense/wan/envs/cog/bin/torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29641 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_sparse_prompt_full_ddp.py" --index_dir /pfs/mayuema/dense/wan/cache/wan22_sparse_coordinates_v1_20261011 --output_dir /pfs/mayuema/dense/wan/outputs/wan22_sparse_prompt_hyperlora_full_4gpu_25k_20261012 --max_steps 25000 --save_steps 500 --eval_steps 1000 --eval_samples 372 --max_hours 0 --stop_file /tmp/wan22_sparse_prompt_full25k_STOP_REQUESTED_20261012
