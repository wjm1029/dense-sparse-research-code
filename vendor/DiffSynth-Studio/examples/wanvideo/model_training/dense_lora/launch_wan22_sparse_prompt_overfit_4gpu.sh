#!/usr/bin/env bash
set -euo pipefail
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export NCCL_SHM_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo
DIR=examples/wanvideo/model_training/dense_lora
/pfs/mayuema/dense/wan/envs/cog/bin/python "$DIR/cog_runtime_bootstrap.py" "$DIR/test_wan22_sparse_prompt.py"
/pfs/mayuema/dense/wan/envs/cog/bin/torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29639 "$DIR/cog_runtime_bootstrap.py" "$DIR/train_wan22_sparse_prompt_ddp.py" --index_dir /pfs/mayuema/dense/wan/cache/wan22_sparse_coordinates_v1_20261011 --output_dir /pfs/mayuema/dense/wan/outputs/wan22_sparse_prompt_hyperlora_overfit_20261012 --overfit_clip auto --max_steps 500 --save_steps 500 --eval_steps 100 --eval_samples 1 --resume /tmp/wan22_sparse_prompt_wan22_sparse_prompt_hyperlora_overfit_20261012/trainer_state_step-0.pt
