# Wan2.1 depth LoRA overfit

This directory implements the first depth-only checkpoint described in
`/pfs/mayuema/dense/wan/docs/wan21_genception_lora_modification_plan.md`.

The trainer caches one clip's source latent, target latent, and fixed prompt
embedding in memory, then optimizes the official DiffSynth LoRA modules with:

```text
MSE(-DiT(z_source, timestep=0, prompt="output: depth"), z_target)
```

It does not add noise, sample timesteps, call the scheduler, or add task
embeddings.  It is deliberately an overfit gate before building the full depth
latent cache and launching the 10k-step dataset run.

Run a one-step smoke test first:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/wan21/bin/python \
  examples/wanvideo/model_training/dense_lora/train_depth_overfit.py \
  --dataset_root /pfs/mayuema/dense/dataset/prod_final_9110_20260816/train \
  --overfit_clip 0a1afa764b9fb300/008f8fd4bbb646de9069 \
  --model_dir /pfs/mayuema/dense/wan/Wan2.1-T2V-1.3B \
  --output_path /pfs/mayuema/dense/wan/outputs/depth_overfit_smoke \
  --gradient_checkpointing \
  --dry_run
```

After it passes, use `Wan2.1-T2V-1.3B-depth-overfit.sh` for the 200-step
single-clip overfit gate.  The later 10k-step full-data run uses a separate
four-GPU launcher.
