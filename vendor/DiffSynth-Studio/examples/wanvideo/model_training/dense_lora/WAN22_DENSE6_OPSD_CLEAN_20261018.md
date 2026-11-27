# Dense6 long477 OPSD, clean-data stage (2026-10-18)

Host sz6670, GPUs 0–3; tmux `dense6_opsd_clean_20261018`.
Source is completed native Dense6 SF4800:
`/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_self_forcing_dispersion_4800_20261017/step-4800.safetensors`.
All six task routes, learned router, LoRA and AR history projection are inherited.
Fresh stage optimizer, EMA initialized from inherited student weights.

## Data filtering

Read `/pfs/mayuema/dense/wan/outputs/long1079_texture_audit_20261018/CLIPS.json`.
Exclude every `broken_color_reference=true` clip across all six tasks, not
just albedo. This removes 426 confirmed broken-asset clips (412 train, 14 val).
Retain 653 clips: 626 train and 27 val, all human. These counts follow the
audited assets, not an unconditional species heuristic or a purple-pixel test.
Original data and latent caches are unchanged. Keep the original deterministic
split; do not move validation clips into training. Audit hashes and exact
retained/excluded IDs are bound into every checkpoint contract.
The audit covers actor materials, not all scene/background rendering defects.
Earlier source weights have seen the excluded data; this stage does not undo
that history or claim to retrain the entire pipeline on clean data.

## Method

5250 optimizer steps = prior Dense OPSD3500 × requested1.5.
Chunk3, 120 latent frames, 40 chunks/update, RGB-only input, six-task cycling,
four distinct clips per global batch. Student history is previous prediction,
detached. Bounded K/V: sink3 + local12 latent frames, absolute positions.
This changes the SF full-prefix context to bounded inference; evaluation must
instantiate `Dense6OPSDModule`, not `LongDenseModule` with full-prefix caches.

EMA teacher decay .9999. Teacher gets the same current RGB, task, prompt and
student-generated previous history. After evaluating chunk k, replay chunk k−1
with clean GT history and replace only that older teacher K/V span. This keeps
the newest generated chunk intact. Teacher forward precedes each student's
forward/backward and bypasses dispersion collectives. No teacher gradients.

Loss = mean latent MSE + .001 router identity + .01 block9 full-feature
dispersion(tau1) + mean EMA-teacher latent MSE after latent frame21.
40 supervised chunks and33 distilled chunks, normalized separately.
This is t=0 task-predictor distillation, not a reproduction of diffusion-flow
distillation. LR2e-5, router2e-6, warmup20, AdamW WD.01, grad clip1.

## Launch, checkpoints and validation

Repo `/pfs/mayuema/dense/wan/DiffSynth-Studio`, script directory
`examples/wanvideo/model_training/dense_lora`.
Run `bash .../launch_dense6_long477_opsd_clean_20261018.sh`.
It runs six real optimizer updates, one per task, followed by a24-entry clean
validation smoke and paired save. Formal run resumes that exact smoke state,
including optimizer, scheduler, EMA, four-rank RNG and next-batch cursor; the
six verified updates count toward5250. Initial formal evaluation then checks
all27 clean validation clips × six tasks under bounded predicted-history
rollout. Full validation every500 updates and final. Student checkpoints
every100 and final; trainer state includes teacher EMA.

Output `/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_opsd_clean_local12_5250_20261018`.
Smoke sibling `wan22_dense6_long477_opsd_clean_smoke6_20261018`.
Logs `/pfs/mayuema/dense/wan/logs/dense6_long477_opsd_clean_smoke6_20261018.log`
and `dense6_long477_opsd_clean5250_20261018.log`.
Checkpoints have native student weights + `condition_contexts`, schema
`wan22_dense6_long477_opsd_clean_v1`; pair with exact runtime_source snapshot.
PFS upload is atomic with bounded local checkpoint spool and8GiB free guard.
Formal resume: same launcher detects `trainer_state_latest.pt` automatically.
`TRAINING_COMPLETE` means all5250 updates and checkpoint uploads completed.

Deployment verified: six task updates had finite nonzero gradients and teacher
distillation. Paired step6 checkpoint was compared tensor-by-tensor with trainer
state:612 student/buffer tensors,610 EMA tensors and610 optimizer states at step6,
four RNG bundles, scheduler6 and next-batch6. Clean/excluded split disjointness
passed. Receipt `SMOKE_VERIFIED.json` in the smoke output. Formal run restored
step6, completed all27×6 clean baseline rollouts and advanced beyond step173.
Observed22.7 seconds/update, driver memory about25021MiB/card.
