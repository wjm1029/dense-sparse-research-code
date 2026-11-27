"""Compare offline, causal-zero-shot, and trained-causal Wan2.2 checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_wan22_causal_ddp import CausalDense4LoRAModule
from train_wan22_dense4_ddp import (
    CachedDense4Dataset,
    Dense4LoRAModule,
    ID_TO_TASK,
    LATENT_SHAPE,
    TASK_TO_ID,
)
from wan22_cache_integrity import sha256_file
from wan22_dense4_checkpoint import (
    build_provenance_metadata,
    checkpoint_metadata,
    load_dense4_checkpoint,
)
from wan22_masked_loss import masked_latent_mse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("offline", "causal_zero_shot", "causal_trained"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--code_patch_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_clips", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--streaming_only", action="store_true")
    parser.add_argument("--compute_dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--model_dir", default="/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B")
    parser.add_argument("--prompt_dir", default="/pfs/mayuema/dense/prompts")
    parser.add_argument(
        "--depth_cache_dir",
        default="/pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40",
    )
    parser.add_argument(
        "--target_cache_dir",
        default="/pfs/mayuema/dense/wan/cache/wan22_dense4_targets_bf16_bright40",
    )
    return parser.parse_args()


def audited_expected_provenance(
    module: Dense4LoRAModule,
    *,
    checkpoint: Path,
    metadata: dict[str, str],
    model_dir: Path,
    prompt_texts: list[str],
    args: argparse.Namespace,
) -> dict[str, str]:
    result = build_provenance_metadata(
        module,
        model_dir=model_dir,
        prompt_texts=prompt_texts,
        prompt_dir=args.prompt_dir,
        depth_cache_dir=args.depth_cache_dir,
        target_cache_dir=args.target_cache_dir,
        code_patch_path=args.code_patch_path,
        repo_dir=THIS_DIR.parents[3],
    )
    patch_hash = sha256_file(Path(args.code_patch_path).resolve())
    if patch_hash != metadata.get("runtime_source_patch_sha256"):
        raise ValueError("Runtime source patch does not match checkpoint metadata")
    # Cross-code evaluation intentionally runs an old offline checkpoint through
    # the new parameter-free causal dispatcher. The immutable old patch remains
    # hash-verified; only its recorded source HEAD is used for strict loading.
    result["diffsynth_git_head"] = metadata["diffsynth_git_head"]
    return result


def per_frame_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> list[float]:
    error = (prediction.float() - target.float()).square()
    weight = mask.float().expand_as(error)
    numerator = (error * weight).sum(dim=(0, 2, 3))
    denominator = weight.sum(dim=(0, 2, 3)).clamp_min(1.0)
    return (numerator / denominator).cpu().tolist()


@torch.inference_mode()
def streaming_equivalence(
    module: CausalDense4LoRAModule,
    source: torch.Tensor,
    task_id: int,
    prompt_id: int,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float | bool | list[float]]:
    device = source.device
    task_ids = torch.tensor([task_id], device=device, dtype=torch.long)
    prompt_ids = torch.tensor([prompt_id], device=device, dtype=torch.long)
    try:
        if module.router_mode == "learned":
            module.task_router(task_ids)
        else:
            module.task_router.activate_identity(task_ids)
        timestep = torch.zeros(1, device=device, dtype=compute_dtype)
        context = module.context_table[prompt_ids].to(dtype=compute_dtype)
        full = module.pipe.model_fn(
            dit=module.pipe.dit,
            latents=source,
            timestep=timestep,
            context=context,
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
            cfg_merge=False,
        )
        caches = module.causal_controller.new_kv_cache()
        chunks = [
            module.causal_controller.streaming_forward(
                module.pipe.dit,
                source[:, :, frame : frame + 1],
                timestep,
                context,
                frame_offset=frame,
                caches=caches,
            )
            for frame in range(LATENT_SHAPE[1])
        ]
        streamed = torch.cat(chunks, dim=2)
    finally:
        module.task_router.clear()
    difference = (full.float() - streamed.float()).abs()
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    reference = float(full.float().abs().mean())
    frame_mean_abs = difference.mean(dim=(0, 1, 3, 4)).cpu().tolist()
    frame_max_abs = difference.amax(dim=(0, 1, 3, 4)).cpu().tolist()
    frame_reference = full.float().abs().mean(dim=(0, 1, 3, 4))
    frame_relative_mean_abs = (
        difference.mean(dim=(0, 1, 3, 4)) / frame_reference.clamp_min(1e-12)
    ).cpu().tolist()
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "mean_abs_relative_to_output": mean_abs / max(reference, 1e-12),
        "allclose_rtol_2e_2_atol_2e_2": bool(torch.allclose(full, streamed, rtol=2e-2, atol=2e-2)),
        "per_frame_mean_abs": frame_mean_abs,
        "per_frame_max_abs": frame_max_abs,
        "per_frame_relative_mean_abs": frame_relative_mean_abs,
    }


def main() -> None:
    args = parse_args()
    if args.num_clips <= 0:
        raise ValueError("num_clips must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Validation requires CUDA")
    torch.cuda.set_device(device)
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.compute_dtype]
    if compute_dtype != torch.bfloat16 and not args.streaming_only:
        raise ValueError("float32 is supported only with --streaming_only")

    checkpoint = Path(args.checkpoint).resolve()
    model_dir = Path(args.model_dir).resolve()
    metadata = checkpoint_metadata(checkpoint)
    prompt_texts = json.loads(metadata["output_prompt_texts"])
    dataset = CachedDense4Dataset(
        args.depth_cache_dir,
        args.target_cache_dir,
        allow_legacy_v1_cache=True,
    )
    if dataset.prompt_texts != prompt_texts:
        raise ValueError("Evaluation prompt table differs from checkpoint prompt table")
    if args.num_clips > dataset.num_clips:
        raise ValueError("num_clips exceeds the cache")
    clip_indices = sorted(random.Random(args.seed).sample(range(dataset.num_clips), args.num_clips))

    module_class = Dense4LoRAModule if args.mode == "offline" else CausalDense4LoRAModule
    module = module_class(
        str(model_dir),
        int(metadata["lora_rank"]),
        metadata["lora_targets"],
        int(metadata["router_hidden_dim"]),
        metadata["router_mode"],
        0.0,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(prompt_texts, device)
    expected_provenance = audited_expected_provenance(
        module,
        checkpoint=checkpoint,
        metadata=metadata,
        model_dir=model_dir,
        prompt_texts=prompt_texts,
        args=args,
    )
    loaded = load_dense4_checkpoint(
        module,
        checkpoint,
        expected_provenance=expected_provenance,
        lora_rank=int(metadata["lora_rank"]),
        lora_targets=metadata["lora_targets"],
        router_hidden_dim=int(metadata["router_hidden_dim"]),
        expected_router_mode=metadata["router_mode"],
    )
    module.pipe.dit.to(device=device, dtype=compute_dtype).eval()
    torch.set_grad_enabled(False)

    if args.streaming_only:
        clip_index = clip_indices[0]
        source, _, task_id, prompt_id, _ = dataset[clip_index * len(TASK_TO_ID)]
        source_batch = source.unsqueeze(0).to(device=device, dtype=compute_dtype)
        stream_check = streaming_equivalence(
            module,
            source_batch,
            task_id,
            prompt_id,
            compute_dtype=compute_dtype,
        )
        result = {
            "schema": "wan22_streaming_drift_v1",
            "mode": args.mode,
            "compute_dtype": args.compute_dtype,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_step": int(loaded["step"]),
            "clip_index": clip_index,
            "clip_id": dataset.clip_ids[clip_index],
            "task": ID_TO_TASK[task_id],
            "streaming_equivalence": stream_check,
        }
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        return

    samples = []
    task_values: dict[str, list[float]] = {task: [] for task in TASK_TO_ID}
    task_frame_values: dict[str, list[list[float]]] = {task: [] for task in TASK_TO_ID}
    stream_check = None
    for clip_index in clip_indices:
        for task_id in range(len(TASK_TO_ID)):
            source, target, actual_task_id, prompt_id, mask = dataset[clip_index * 4 + task_id]
            if actual_task_id != task_id:
                raise RuntimeError("Dataset task ordering changed")
            source_batch = source.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            try:
                velocity, _ = module.predict_velocity(
                    source_batch,
                    torch.tensor([task_id], device=device, dtype=torch.long),
                    torch.tensor([prompt_id], device=device, dtype=torch.long),
                )
            finally:
                module.task_router.clear()
            prediction = -velocity[0]
            target_device = target.to(device=device, dtype=torch.bfloat16)
            mask_device = mask.to(device=device, dtype=torch.float32)
            loss = float(
                masked_latent_mse(
                    prediction.unsqueeze(0),
                    target_device.unsqueeze(0),
                    mask_device.unsqueeze(0),
                )
            )
            frames = per_frame_mse(prediction, target_device, mask_device)
            task = ID_TO_TASK[task_id]
            task_values[task].append(loss)
            task_frame_values[task].append(frames)
            samples.append(
                {
                    "clip_index": clip_index,
                    "clip_id": dataset.clip_ids[clip_index],
                    "task": task,
                    "loss": loss,
                    "per_frame_mse": frames,
                }
            )
            if args.mode == "causal_trained" and stream_check is None:
                stream_check = streaming_equivalence(
                    module, source_batch, task_id, prompt_id, compute_dtype=compute_dtype
                )

    task_metrics = {}
    all_losses = []
    for task in TASK_TO_ID:
        losses = task_values[task]
        all_losses.extend(losses)
        frame_rows = task_frame_values[task]
        task_metrics[task] = {
            "mean_mse": sum(losses) / len(losses),
            "per_frame_mean_mse": [
                sum(row[frame] for row in frame_rows) / len(frame_rows)
                for frame in range(LATENT_SHAPE[1])
            ],
        }
    result = {
        "schema": "wan22_causal_three_way_validation_v1",
        "mode": args.mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(loaded["step"]),
        "attention": "offline_bidirectional" if args.mode == "offline" else "block_causal_framewise",
        "num_clips": len(clip_indices),
        "clip_indices": clip_indices,
        "overall_mean_mse": sum(all_losses) / len(all_losses),
        "task_metrics": task_metrics,
        "streaming_equivalence": stream_check,
        "samples": samples,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("mode", "checkpoint_step", "overall_mean_mse", "streaming_equivalence")}), flush=True)


if __name__ == "__main__":
    main()
