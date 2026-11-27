"""Wan2.2 t=0 post-training with OPSD-style on-policy prefix distillation.

This is intentionally a separate entry point.  OPSD-V trains a multi-step
diffusion flow; our dense task checkpoint is a single-forward t=0 predictor, so
we transfer its generated-student/clean-teacher cache policy and EMA teacher,
while retaining the existing supervised task loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

import train_wan22_dense4_ddp as dense4
import train_wan22_long477_ar_self_forcing_ddp as sf
from wan22_ar_self_forcing import detach_kv_cache, training_streaming_forward
from wan22_ar_teacher_forcing import combine_rgb_and_history, install_ar_history_adapter
from wan22_cache_integrity import sha256_file
from wan22_dense4_checkpoint import (
    OPTIONAL_PROVENANCE_KEYS,
    REQUIRED_PROVENANCE_KEYS,
    checkpoint_metadata,
    load_dense4_checkpoint,
    save_dense4_checkpoint,
)
from wan22_masked_loss import masked_latent_mse
from wan22_causal_attention import install_causal_attention
from wan22_opsd_bounded_causal_attention import install_opsd_bounded_causal_attention
from wan22_opsd_prefix_cache import (
    append_cached_chunk_, clone_prefix, extract_latest_chunk,
    replace_chunk_from_replay,
    swap_trainable_parameters_,
    truncate_full_cache_,
    update_ema_,
)
from wan22_task_router import ID_TO_TASK


THIS_DIR = Path(__file__).resolve().parent
UPSTREAM_OPSD_COMMIT = "8ecb193fea884efdbea6e731498ce0853cae180b"


class OPSDPrefixModule(dense4.Dense4LoRAModule):
    def __init__(self, *args, attention_context: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if attention_context == "full":
            self.causal_controller = install_causal_attention(
                self.pipe.dit, latent_shape=sf.LATENT_SHAPE,
                frames_per_chunk=sf.CHUNK, local_attn_frames=-1, sink_frames=0,
            )
        elif attention_context == "upstream_local":
            self.causal_controller = install_opsd_bounded_causal_attention(
                self.pipe.dit, latent_shape=sf.LATENT_SHAPE,
                frames_per_chunk=sf.CHUNK, local_attn_frames=12, sink_frames=3,
            )
        else:
            raise ValueError(f"unsupported attention_context={attention_context}")
        self.ar_controller = install_ar_history_adapter(
            self.pipe.dit, self.causal_controller, sf.AR_CONFIG
        )

    def forward(self, rgb, history, target, mask, task_ids, prompt_ids, frame_offset):
        if self.context_table is None:
            raise RuntimeError("Prompt table is not ready")
        gates = self.task_router(task_ids)
        timestep = torch.zeros(task_ids.shape[0], device=task_ids.device, dtype=torch.bfloat16)
        ar_input = combine_rgb_and_history(rgb, history, sf.AR_CONFIG)
        prediction = -training_streaming_forward(
            self.causal_controller, self.pipe.dit, ar_input, timestep,
            self.context_table[prompt_ids], frame_offset=int(frame_offset),
            caches=self._active_caches,
        )
        data_loss = (
            masked_latent_mse(prediction, target, mask)
            if bool(mask.any()) else prediction.float().sum() * 0.0
        )
        identity_loss = self.task_router.identity_regularization(gates)
        return prediction, data_loss, identity_loss, gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_cache_dir", required=True)
    parser.add_argument("--main_normal_cache_dir", required=True)
    parser.add_argument("--lingbot_depth_cache_dir", required=True)
    parser.add_argument("--lingbot_normal_cache_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--prompt_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--code_patch_path", required=True)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--init_code_patch_path", required=True)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--router_learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--router_identity_weight", type=float, default=1e-3)
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--loss_start_frame", type=int, default=21)
    parser.add_argument("--attention_context", choices=("full", "upstream_local"), default="full")
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--single_gpu_smoke", action="store_true")
    return parser.parse_args()


def copy_parameters_(named_parameters, values) -> None:
    with torch.no_grad():
        for name, parameter in named_parameters:
            parameter.copy_(values[name])


def teacher_forward(
    module, source, target, mask, task_ids, prompt_ids, history, offset,
    teacher_cache, ema, named_parameters,
):
    """Evaluate EMA teacher, then clean the second-latest cache chunk."""

    backup = swap_trainable_parameters_(named_parameters, ema)
    try:
        with torch.no_grad():
            module._active_caches = teacher_cache
            teacher_prediction, _, _, _ = module.task_router.call_with_error_cleanup(
                lambda: module(
                    source[:, :, offset:offset + sf.CHUNK], history,
                    target[:, :, offset:offset + sf.CHUNK],
                    mask[:, :, offset:offset + sf.CHUNK], task_ids, prompt_ids, offset,
                )
            )
            module.task_router.clear()

            if offset >= sf.CHUNK:
                replace_offset = offset - sf.CHUNK
                start_token = replace_offset * module.causal_controller.config.tokens_per_frame
                clean_history = (
                    torch.zeros_like(history)
                    if replace_offset == 0
                    else target[:, :, replace_offset - sf.CHUNK:replace_offset]
                )
                if getattr(teacher_cache[0], "positions", None) is None:
                    latest = extract_latest_chunk(
                        teacher_cache, module.causal_controller.config.chunk_tokens
                    )
                    truncate_full_cache_(teacher_cache, start_token)
                    module._active_caches = teacher_cache
                    module.task_router.call_with_error_cleanup(
                        lambda: module(
                            source[:, :, replace_offset:replace_offset + sf.CHUNK], clean_history,
                            target[:, :, replace_offset:replace_offset + sf.CHUNK],
                            mask[:, :, replace_offset:replace_offset + sf.CHUNK],
                            task_ids, prompt_ids, replace_offset,
                        )
                    )
                    module.task_router.clear()
                    append_cached_chunk_(
                        teacher_cache, latest,
                        expected_start=start_token + module.causal_controller.config.chunk_tokens,
                    )
                else:
                    replay = clone_prefix(teacher_cache, start_token)
                    module._active_caches = replay
                    module.task_router.call_with_error_cleanup(
                        lambda: module(
                            source[:, :, replace_offset:replace_offset + sf.CHUNK], clean_history,
                            target[:, :, replace_offset:replace_offset + sf.CHUNK],
                            mask[:, :, replace_offset:replace_offset + sf.CHUNK],
                            task_ids, prompt_ids, replace_offset,
                        )
                    )
                    module.task_router.clear()
                    replace_chunk_from_replay(
                        teacher_cache, replay, start_token=start_token,
                        chunk_tokens=module.causal_controller.config.chunk_tokens,
                    )
        return teacher_prediction
    finally:
        copy_parameters_(named_parameters, backup)
        module.task_router.clear()


def save_checkpoint(module, output, step, args, provenance) -> None:
    init = Path(args.init_checkpoint).resolve()
    save_dense4_checkpoint(
        module, output / f"step-{step}.safetensors", step=step,
        world_size=dist.get_world_size(), lora_rank=32,
        lora_targets="q,k,v,o,ffn.0,ffn.2", router_hidden_dim=512,
        router_mode="learned", provenance_metadata=provenance,
        extra_metadata={
            "initialization_checkpoint": str(init),
            "initialization_checkpoint_sha256": sha256_file(init),
            "causal_attention": "block_prefix_v1", "causal_num_frames": "120",
            "causal_tokens_per_frame": "390", "causal_frames_per_chunk": "3",
            "causal_local_attn_frames": "-1" if args.attention_context == "full" else "12",
            "causal_sink_frames": "0" if args.attention_context == "full" else "3",
            "causal_streaming_kv": "per_layer_v1" if args.attention_context == "full" else "per_layer_bounded_absolute_positions_v1",
            "opsd_attention_context": args.attention_context, "ar_teacher_forcing": "none",
            "ar_inference_history": "previous_predicted_chunk_v1",
            "ar_num_frames": "120", "ar_frames_per_chunk": "3", "ar_num_chunks": "40",
            "ar_latent_channels": "48", "self_forcing": "previous_predicted_chunk_stopgrad_v1",
            "bptt": "chunkwise_truncated_stopgrad_kv_and_history_v1",
            "post_training": "opsd_prefix_distillation_t0_v1",
            "opsd_upstream_commit": UPSTREAM_OPSD_COMMIT,
            "opsd_student_context": "generated_kv",
            "opsd_teacher_context": "gt_kv_delayed_one_chunk",
            "opsd_teacher_trajectory": "student",
            "opsd_teacher_weights": f"ema_{args.ema_decay}",
            "opsd_loss": "masked_x0_plus_supervised_t0",
            "opsd_loss_start_frame": str(args.loss_start_frame),
            "opsd_distill_weight": str(args.distill_weight),
            "normal_coordinate_contract": "camera_x_right_y_up_z_toward_camera_v1",
        },
    )


def main() -> None:
    args = parse_args()
    if args.loss_start_frame < 0 or args.loss_start_frame % sf.CHUNK:
        raise ValueError("loss_start_frame must be a non-negative chunk boundary")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4 and not (world == 1 and args.single_gpu_smoke):
        raise ValueError("Formal OPSD-prefix post-training requires four GPUs")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    init_meta = checkpoint_metadata(args.init_checkpoint)
    if init_meta.get("self_forcing") != "previous_predicted_chunk_stopgrad_v1":
        raise ValueError("Initialization must be the audited AR self-forcing checkpoint")
    if init_meta.get("ar_frames_per_chunk") != "3":
        raise ValueError("Initialization must use three-frame latent chunks")
    if sha256_file(args.init_code_patch_path) != init_meta["runtime_source_patch_sha256"]:
        raise ValueError("Initialization source patch hash mismatch")
    prompt_texts = json.loads(init_meta["output_prompt_texts"])
    dataset = sf.Long477Dataset(
        Path(args.main_cache_dir).resolve(), Path(args.main_normal_cache_dir).resolve(),
        Path(args.lingbot_depth_cache_dir).resolve(),
        Path(args.lingbot_normal_cache_dir).resolve(), prompt_texts,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    module = OPSDPrefixModule(
        args.model_dir, 32, "q,k,v,o,ffn.0,ffn.2", 512, "learned",
        args.router_identity_weight, attention_context=args.attention_context,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(prompt_texts, device)
    init_provenance = {key: init_meta[key] for key in REQUIRED_PROVENANCE_KEYS}
    for key in OPTIONAL_PROVENANCE_KEYS:
        if key in init_meta:
            init_provenance[key] = init_meta[key]
    load_dense4_checkpoint(
        module, args.init_checkpoint, expected_provenance=init_provenance,
        lora_rank=32, lora_targets="q,k,v,o,ffn.0,ffn.2",
        router_hidden_dim=512, expected_router_mode="learned",
    )
    module.pipe.dit.to(device).train()
    named_parameters = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    ema = {name: p.detach().clone() for name, p in named_parameters}
    ddp = DistributedDataParallel(module, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    groups = dense4.optimizer_param_groups(module, argparse.Namespace(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        router_learning_rate=args.router_learning_rate, router_weight_decay=args.weight_decay,
    ))
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps))
    )
    output = Path(args.output_path).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()) and not args.resume_from:
            raise FileExistsError(f"Refusing non-empty output: {output}")
    dist.barrier()
    manifests = (
        sf.verified_manifest(Path(args.main_cache_dir), sf.MAIN_SCHEMA)[0],
        sf.verified_manifest(Path(args.main_normal_cache_dir), sf.MAIN_NORMAL_SCHEMA)[0],
        sf.verified_manifest(Path(args.lingbot_depth_cache_dir), sf.LINGBOT_SCHEMA)[0],
        sf.verified_manifest(Path(args.lingbot_normal_cache_dir), sf.LINGBOT_NORMAL_SCHEMA)[0],
    )
    provenance = sf.provenance_from_caches(args, init_meta, *manifests)
    step = epoch = 0
    if args.resume_from:
        state = torch.load(Path(args.resume_from) / "trainer_state_latest.pt", map_location="cpu", weights_only=False)
        step, epoch = int(state["step"]), int(state["epoch"])
        load_dense4_checkpoint(
            module, Path(args.resume_from) / f"step-{step}.safetensors",
            expected_provenance=provenance, lora_rank=32,
            lora_targets="q,k,v,o,ffn.0,ffn.2", router_hidden_dim=512,
            expected_router_mode="learned",
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema = {name: value.to(device=p.device, dtype=p.dtype) for (name, p), value in zip(named_parameters, (state["teacher_ema"][n] for n, _ in named_parameters))}
    if rank == 0 and not args.resume_from:
        payload = vars(args) | {
            "dataset_samples": len(dataset), "latent_shape": sf.LATENT_SHAPE,
            "chunks": sf.NUM_CHUNKS, "method": "opsd_prefix_distillation_t0_v1",
            "upstream_opsd_commit": UPSTREAM_OPSD_COMMIT,
        }
        (output / "training_args.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (output / "loss.csv").open("w", newline="") as handle:
            csv.writer(handle).writerow([
                "step", "loss", "data_loss", "distill_loss", "identity_loss",
                "task_id", "task", "grad_norm", "seconds", "max_memory_gib",
            ])

    started = time.time()
    while step < args.max_train_steps:
        sampler.set_epoch(epoch)
        for source, target, task_ids, prompt_ids, mask, sample_ids in loader:
            source = source.to(device, dtype=torch.bfloat16, non_blocking=True)
            target = target.to(device, dtype=torch.bfloat16, non_blocking=True)
            mask = mask.to(device, dtype=torch.float32, non_blocking=True)
            task_ids = task_ids.to(device, dtype=torch.long)
            prompt_ids = prompt_ids.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            student_cache = module.causal_controller.new_kv_cache()
            teacher_cache = module.causal_controller.new_kv_cache()
            history = torch.zeros_like(source[:, :, :sf.CHUNK])
            valid = [
                bool(mask[:, :, offset:offset + sf.CHUNK].any())
                for offset in range(0, sf.LATENT_SHAPE[1], sf.CHUNK)
            ]
            valid_task_chunks = sum(valid)
            valid_distill_chunks = sum(
                is_valid and offset >= args.loss_start_frame
                for is_valid, offset in zip(valid, range(0, sf.LATENT_SHAPE[1], sf.CHUNK))
            )
            if valid_task_chunks == 0 or valid_distill_chunks == 0:
                raise ValueError(f"Sample has no valid supervised OPSD chunks: {sample_ids}")
            data_sum = distill_sum = identity_sum = 0.0
            for chunk_index, offset in enumerate(range(0, sf.LATENT_SHAPE[1], sf.CHUNK)):
                teacher_prediction = teacher_forward(
                    module, source, target, mask, task_ids, prompt_ids, history,
                    offset, teacher_cache, ema, named_parameters,
                )
                module._active_caches = student_cache
                sync = nullcontext() if chunk_index == sf.NUM_CHUNKS - 1 else ddp.no_sync()
                with sync:
                    prediction, data_loss, identity_loss, _ = module.task_router.call_with_error_cleanup(
                        lambda o=offset: ddp(
                            source[:, :, o:o + sf.CHUNK], history,
                            target[:, :, o:o + sf.CHUNK], mask[:, :, o:o + sf.CHUNK],
                            task_ids, prompt_ids, o,
                        )
                    )
                    if valid[chunk_index] and offset >= args.loss_start_frame:
                        distill_loss = masked_latent_mse(
                            prediction, teacher_prediction.detach(),
                            mask[:, :, offset:offset + sf.CHUNK],
                        )
                    else:
                        distill_loss = prediction.float().sum() * 0.0
                    loss = (
                        data_loss / valid_task_chunks
                        + args.distill_weight * distill_loss / valid_distill_chunks
                        + args.router_identity_weight * identity_loss / sf.NUM_CHUNKS
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss chunk={chunk_index}")
                    loss.backward()
                data_sum += float(data_loss.detach())
                distill_sum += float(distill_loss.detach())
                identity_sum += float(identity_loss.detach())
                history = prediction.detach()
                detach_kv_cache(student_cache)
                module.task_router.clear()
            finite = torch.tensor(int(dense4.gradients_are_finite(module)), device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite):
                raise FloatingPointError("non-finite gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in module.parameters() if p.requires_grad],
                args.max_grad_norm, error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            update_ema_(ema, named_parameters, args.ema_decay)
            step += 1
            record = torch.tensor([
                data_sum / valid_task_chunks,
                distill_sum / valid_distill_chunks,
                identity_sum / sf.NUM_CHUNKS,
                float(task_ids.item()), float(grad_norm),
            ], device=device)
            gathered = [torch.empty_like(record) for _ in range(world)]
            dist.all_gather(gathered, record)
            if rank == 0:
                elapsed = time.time() - started
                with (output / "loss.csv").open("a", newline="") as handle:
                    writer = csv.writer(handle)
                    for row in gathered:
                        task_id = int(row[3])
                        total = float(row[0]) + args.distill_weight * float(row[1]) + args.router_identity_weight * float(row[2])
                        writer.writerow([step, total, float(row[0]), float(row[1]), float(row[2]), task_id, ID_TO_TASK[task_id], float(row[4]), elapsed, torch.cuda.max_memory_allocated(device) / 1024**3])
                print(
                    f"step={step}/{args.max_train_steps} "
                    f"task={sum(float(row[0]) for row in gathered) / world:.6f} "
                    f"distill={sum(float(row[1]) for row in gathered) / world:.6f} "
                    f"sec_per_step={elapsed / step:.2f} "
                    f"max_mem_gib={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}",
                    flush=True,
                )
            if step % args.save_steps == 0 or step == args.max_train_steps:
                dist.barrier()
                if rank == 0:
                    save_checkpoint(module, output, step, args, provenance)
                    torch.save({
                        "step": step, "epoch": epoch,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "teacher_ema": {name: value.cpu() for name, value in ema.items()},
                    }, output / "trainer_state_latest.pt")
                dist.barrier()
            if step >= args.max_train_steps:
                break
        epoch += 1
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
