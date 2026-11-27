"""Resident-model batch driver for the verified Wan2.2 Dense4 inference path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from infer_wan22_dense4 import (
    DEFAULT_ROOT,
    LATENT_SHAPE,
    Dense4LoRAModule,
    TASK_TO_ID,
    build_provenance_metadata,
    checkpoint_metadata,
    decode_video,
    encode_video,
    load_dense4_checkpoint,
    load_official_vae,
    prepare_output_dir,
    prompt_id_for_task,
    read_video,
    selected_tasks,
    sha256_file,
    write_task_outputs,
)


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--shard_rank", type=int, required=True)
    parser.add_argument("--shard_count", type=int, required=True)
    parser.add_argument("--tasks", default="depth,normal")
    parser.add_argument("--matte_subject", default="person")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_ROOT / "outputs/wan22_dense4_hyperlora/step-10000.safetensors"),
    )
    parser.add_argument("--model_dir", default=str(DEFAULT_ROOT / "Wan2.2-TI2V-5B"))
    parser.add_argument("--official_repo", default=str(DEFAULT_ROOT / "Wan2.2"))
    parser.add_argument("--prompt_dir", default="/pfs/mayuema/dense/prompts")
    parser.add_argument(
        "--depth_cache_dir",
        default=str(DEFAULT_ROOT / "cache/wan22_depth_latents_bf16_bright40"),
    )
    parser.add_argument(
        "--target_cache_dir",
        default=str(DEFAULT_ROOT / "cache/wan22_dense4_targets_bf16_bright40"),
    )
    parser.add_argument(
        "--code_patch_path", default=str(DEFAULT_ROOT / "docs/wan22_dense4_runtime_source.patch")
    )
    parser.add_argument("--min_rgb_mean", type=float, default=40.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_rank < args.shard_count:
        raise ValueError("Invalid shard rank/count")
    tasks = selected_tasks(args.tasks)
    cases = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))["cases"]
    if len(cases) != 20:
        raise ValueError(f"Expected exactly 20 selected cases, got {len(cases)}")
    cases = cases[args.shard_rank :: args.shard_count]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Wan2.2 Dense4 inference requires CUDA")
    torch.cuda.set_device(device)
    checkpoint = Path(args.checkpoint).resolve()
    model_dir = Path(args.model_dir).resolve()
    metadata = checkpoint_metadata(checkpoint)
    prompt_texts = json.loads(metadata["output_prompt_texts"])
    prompt_ids = {
        task: prompt_id_for_task(prompt_texts, task, args.matte_subject) for task in tasks
    }

    module = Dense4LoRAModule(
        str(model_dir),
        int(metadata["lora_rank"]),
        metadata["lora_targets"],
        int(metadata["router_hidden_dim"]),
        metadata["router_mode"],
        0.0,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(prompt_texts, device)
    module.provenance_metadata = build_provenance_metadata(
        module,
        model_dir=model_dir,
        prompt_texts=prompt_texts,
        prompt_dir=args.prompt_dir,
        depth_cache_dir=args.depth_cache_dir,
        target_cache_dir=args.target_cache_dir,
        code_patch_path=args.code_patch_path,
        repo_dir=THIS_DIR.parents[3],
    )
    # Evaluation-only utilities may be added after training without changing
    # the audited model implementation.  Validate the runtime sources against
    # the revision recorded by the checkpoint, as the benchmark driver does.
    module.provenance_metadata["diffsynth_git_head"] = metadata["diffsynth_git_head"]
    module.pipe.dit.to(device).eval()
    loaded_metadata = load_dense4_checkpoint(
        module,
        checkpoint,
        expected_provenance=module.provenance_metadata,
        lora_rank=int(metadata["lora_rank"]),
        lora_targets=metadata["lora_targets"],
        router_hidden_dim=int(metadata["router_hidden_dim"]),
        expected_router_mode=metadata["router_mode"],
    )
    vae = load_official_vae(Path(args.official_repo).resolve(), model_dir, device)
    checkpoint_hash = sha256_file(checkpoint)
    print(
        json.dumps(
            {
                "resident_model_ready": True,
                "shard_rank": args.shard_rank,
                "cases": [case["case_name"] for case in cases],
            }
        ),
        flush=True,
    )

    for case in cases:
        source_path = Path(case["path"]).resolve()
        output_dir = output_root / case["case_name"]
        if (output_dir / "INFERENCE_COMPLETE").is_file():
            print(json.dumps({"skip_complete": case["case_name"]}), flush=True)
            continue
        prepare_output_dir(output_dir, overwrite=True)
        input_frames, detected_fps = read_video(source_path)
        rgb_mean = float(input_frames.mean())
        if rgb_mean < args.min_rgb_mean:
            raise ValueError(
                f"Input RGB mean {rgb_mean:.3f} is below threshold {args.min_rgb_mean}: {source_path}"
            )
        source_latent = encode_video(vae, input_frames, device)
        rows = []
        with torch.inference_mode():
            for task in tasks:
                task_id = TASK_TO_ID[task]
                try:
                    velocity, gates = module.predict_velocity(
                        source_latent.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                        torch.tensor([task_id], device=device, dtype=torch.long),
                        torch.tensor([prompt_ids[task]], device=device, dtype=torch.long),
                    )
                finally:
                    module.task_router.clear()
                prediction_latent = (-velocity[0]).detach().to(
                    device="cpu", dtype=torch.bfloat16
                ).contiguous()
                if tuple(prediction_latent.shape) != LATENT_SHAPE or not torch.isfinite(
                    prediction_latent
                ).all():
                    raise ValueError(f"Invalid predicted {task} latent")
                prediction = decode_video(vae, prediction_latent, device)
                write_task_outputs(
                    output_dir,
                    task,
                    input_frames,
                    prediction,
                    prediction_latent,
                    detected_fps,
                )
                rows.append(
                    {
                        "task": task,
                        "task_id": task_id,
                        "prompt": prompt_texts[prompt_ids[task]],
                        "gate_min": float(gates.min()),
                        "gate_max": float(gates.max()),
                        "gate_mean": float(gates.mean()),
                    }
                )
                del velocity, gates, prediction_latent, prediction
                torch.cuda.empty_cache()
        case_manifest = {
            "schema": "wan22_dense4_single_forward_inference_v1",
            "resident_batch_driver": True,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_step": int(loaded_metadata["step"]),
            "base_model": str(model_dir),
            "vae_checkpoint": str(model_dir / "Wan2.2_VAE.pth"),
            "source": str(source_path),
            "source_kind": "video",
            "source_rgb_mean": rgb_mean,
            "source_latent_shape": list(source_latent.shape),
            "fps": detected_fps,
            "tasks": rows,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(case_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "INFERENCE_COMPLETE").write_text(
            json.dumps({"checkpoint_step": int(loaded_metadata["step"]), "tasks": tasks}),
            encoding="utf-8",
        )
        print(json.dumps({"completed": case["case_name"], "tasks": tasks}), flush=True)
        del input_frames, source_latent
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
