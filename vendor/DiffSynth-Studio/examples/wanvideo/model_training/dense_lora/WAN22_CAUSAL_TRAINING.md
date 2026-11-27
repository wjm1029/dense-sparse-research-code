# Wan2.2 causal Stage-1 baseline

This baseline continues from:

`/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_lingbot_stage2_20k/step-20000.safetensors`

It changes only temporal self-attention. The existing `t=0` single-forward
training path, cached latents, prompt encoder outputs, four task losses,
HyperLoRA routing, optimizer groups, and checkpoint tensors remain unchanged.

## Attention contract

For latent layout `[48, 21, 30, 52]` and patch size `[1, 2, 2]`, every latent
frame contains 390 DiT tokens. With the default one-frame chunks, tokens in
frame `t` attend to every token in frames `0..t`, including bidirectional
attention inside frame `t`, and cannot attend to later frames.

`wan22_causal_attention.py` replaces only the parameter-free attention
dispatcher under each self-attention layer. It verifies that installation does
not change any state-dict key. Q/K/V/O PEFT modules and the 300-site task router
therefore keep the checkpoint naming and runtime hooks used by the offline
model.

The same module provides inference-only `streaming_forward`. It applies temporal
RoPE at the global frame offset and stores RoPE-applied K/V tensors separately
for all 30 layers. The test suite requires full causal single-forward and
streaming cached execution to agree numerically.

## Run

From the repository root:

```bash
MAX_TRAIN_STEPS=10000 \
  examples/wanvideo/model_training/dense_lora/train_wan22_causal_4gpu.sh
```

The launcher refuses a dirty tracked worktree, exports the current causal commit
as the immutable runtime patch, and embeds causal layout fields plus the exact
initialization checkpoint hash in every saved checkpoint.

Initialization is an intentional cross-code transition. The causal wrapper
requires the old immutable source patch hash to match the input checkpoint,
then uses that checkpoint's recorded Git HEAD only for loading. All other base
model, cache, prompt, LoRA-site, router, and tensor checks stay strict. New
checkpoints bind the causal commit HEAD and causal runtime patch.

Optional environment variables:

- `WAN_CAUSAL_FRAMES_PER_CHUNK` (default `1`)
- `WAN_CAUSAL_LOCAL_ATTN_FRAMES` (default `-1`, global prefix)
- `WAN_CAUSAL_SINK_FRAMES` (default `0`)
- `OUTPUT_PATH`, `MAX_TRAIN_STEPS`, `MASTER_PORT`

`21` must be divisible by `WAN_CAUSAL_FRAMES_PER_CHUNK`; consequently the
current 21-frame training cache supports chunk sizes 1, 3, 7, or 21.

## Validation

Use the cog runtime and the existing bootstrap:

```bash
PYTHONPATH="$(pwd):$(pwd)/examples/wanvideo/model_training/dense_lora" \
  /pfs/mayuema/dense/wan/envs/cog/bin/python \
  examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py \
  examples/wanvideo/model_training/dense_lora/test_wan22_causal_attention.py
```

The tests cover causal isolation, exact attention-level full/streaming equality,
end-to-end tiny Wan equality with global temporal RoPE, unchanged state-dict
keys, and autograd.
