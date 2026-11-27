# Wan2.2 OPSD-style prefix post-training

## Scope

This implementation transfers the on-policy context-distillation mechanism
from OPSD-V commit `8ecb193fea884efdbea6e731498ce0853cae180b` into the existing Wan2.2 dense
task model.  It does not copy or modify the official Wan2.2 implementation.

The upstream OPSD-V model is a few-step diffusion generator and distills flow
predictions at several noisy timesteps.  Our checkpoint is a single-forward
`t=0` RGB-to-task predictor.  Therefore this adaptation intentionally keeps the
existing masked latent task loss and adds masked `x0` teacher/student
distillation at `t=0`; it is not described as a reproduction of upstream flow
distillation.

## Context policy

- Student weights: trainable Wan2.2 LoRA/router/history adapter.
- Teacher weights: EMA of all trainable post-training parameters.
- Student history and KV: generated trajectory throughout.
- Teacher current query: same RGB, generated previous chunk, temporal position,
  and task prompt as the student.
- Teacher older KV: the latest generated chunk is retained.  Once the following
  chunk has been evaluated, the second-latest chunk is replayed using clean GT
  history and its K/V span replaces the generated span.
- Loss starts at latent frame 21, matching the upstream burn-in convention.
- Chunk size is 3 latent frames, matching the existing Wan2.2 AR checkpoint and
  the upstream default.
- `--attention_context full` is the quality-first default because the input
  checkpoint was trained with full-prefix attention.  Its in-place replacement
  retains only one extra generated chunk during teacher replay.  The verified
  120-frame smoke fits the current 98 GiB cards, though allocator-reserved
  memory leaves less headroom than the local-window ablation.
- `--attention_context upstream_local` retains a 3-frame sink and 12-frame
  sliding window, matching the upstream OPSD-V defaults.  It fits the current
  host and is useful as an ablation, but changes the checkpoint's context
  distribution.  It does not change earlier training or inference entry points.

## Provenance and launch

The upstream repository is stored separately at
`/pfs/mayuema/dense/wan/third_party/OPSD-V`.  Its Git commit is recorded in each
checkpoint.  The launch script records the current adaptation commit as an
immutable multi-commit email patch (from the audited pre-adaptation base through
`HEAD`) before training.

Run a one-step deployment smoke test by setting `MAX_TRAIN_STEPS=1`, a fresh
`OUTPUT_PATH`, and invoking `train_wan22_long477_opsd_prefix_4gpu.sh`.

## Verified deployment smoke

On 2026-09-15, commit `5851251` completed a four-GPU, one-step
`upstream_local` smoke run from the clean self-forcing `step-3200` checkpoint:

- 40 latent chunks completed with separate teacher/student caches, backward,
  optimizer update, EMA update, and checkpoint save.
- Mean task loss: `0.068776`; mean prefix-distillation loss: `0.000434`.
- Runtime: `24.88 s/step`; peak allocated memory: `24.03 GiB/GPU`.
- Checkpoint SHA-256: `09da381160f7b80c6ea74453aacaeff27e64cbdf5555a290875a9b973371f18a`.
- Runtime source patch SHA-256 recorded by the checkpoint:
  `281efb1ab2393ced922a31a03983fa95847039c63e8b756150dc533b9c1f5344`.

The local-window smoke output is at
`/pfs/mayuema/dense/wan/outputs/wan22_long477_opsd_prefix_t0_smoke_local12`.
This verifies that ablation's deployment path, not model quality.  Formal
quality training should use the default full-prefix path.

The quality-first `full` path was then verified separately on the same four
98 GiB GPUs:

- Mean task loss: `0.065081`; mean prefix-distillation loss: `0.000529`.
- Runtime: `29.86 s/step`; peak PyTorch allocated memory: `52.34 GiB/GPU`.
  Driver-reported process memory reached approximately `84.1 GiB/GPU` during
  rollout and returned to zero after clean completion.
- Checkpoint SHA-256: `69c1b349c54f547708cf39f3fb4a06992918918dfa31b8cac2d1d7ba48e8891c`.
- Smoke output:
  `/pfs/mayuema/dense/wan/outputs/wan22_long477_opsd_prefix_t0_smoke_full`.

The full-prefix smoke demonstrates that the current host can run the default
post-training path; a larger-memory GPU is optional rather than required.
