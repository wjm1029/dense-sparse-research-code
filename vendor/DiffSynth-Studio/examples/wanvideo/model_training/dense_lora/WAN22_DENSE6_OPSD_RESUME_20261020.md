# Dense6 clean OPSD resume (2026-10-20)

Previous run exited2026-10-19 at2267 while appending PFS train_metrics.jsonl
with ENOSPC. Latest complete paired checkpoint is2200, not2267 or5250.
Resume optimizer/scheduler,610 teacher EMA tensors, four-rank RNG and epoch/batch
cursor from:
`/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_opsd_clean_local12_5250_20261018/trainer_state_latest.pt`.
Unsaved2201–2267 updates are replayed. Target remains5250 total (3050 updates
remaining at launch). No new optimizer or LoRA initialization.

Host sz6670 GPUs0–3, tmux `dense6_opsd_resume_20261020`.
New output:
`/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_opsd_clean_local12_5250_resume_20261020`.
Original interrupted output and stdout log retained. New stdout log:
`/pfs/mayuema/dense/wan/logs/dense6_opsd_resume_20261020.log`.

Unchanged training entry point and hash-bound contract, native SF4800 source,
sink3/local12, six tasks, loss/dispersion/learning rates,626 clean train clips and
27 clean holdouts. All426 broken-color-reference clips remain excluded.

`preflight_dense6_opsd_resume_20261020.py` checks paired student tensors against
safetensors, finite EMA, optimizer steps/scheduler, RNG count, runtime source
hashes, exact clean split counts, and a1MiB PFS write+fsync+atomic rename probe.
Receipt is `RESUME_PREFLIGHT_step2200_20261020.json` in new output.
Launcher `resume_dense6_opsd_clean_20261020.sh` uses the new output's own latest
trainer state on future invocation if one exists; otherwise the old2200 state.
Use the same repo cwd and cog bootstrap as the original stage. Training emits
`DENSE6_OPSD_INITIALIZED step=2200` then starts at2201.

Latest completed validation is step2200,27 holdouts,162 task metrics, two decoded
human cases. Artifact:
`/pfs/mayuema/dense/wan/outputs/wan22_dense6_opsd2200_validation_20261020`.
Local desktop folder `dense6_OPSD2200_验证集_20261020`.
Raymap late-quarter MSE remains elevated; do not claim final model completion
or monotonic validation improvement. Final training completion requires5250
and successfully uploaded paired checkpoint/`TRAINING_COMPLETE`.

Deployment verified: preflight PASS (612 paired tensors,610 EMA/optimizer states,
scheduler2200,four-rank RNG,write/fsync/rename). New run restored2200 and completed
2201 with finite gradients; its MSE0.0606004782 and distillation0.00306605594 match
the original interrupted run's2201 exactly. Four GPUs each use about25019MiB.
