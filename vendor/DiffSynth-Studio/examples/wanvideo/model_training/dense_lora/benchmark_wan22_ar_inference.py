#!/usr/bin/env python3
"""Benchmark resident Wan2.2 chunkwise free-running AR inference only."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from infer_wan22_dense4 import (
    decode_video,
    encode_video,
    load_official_vae,
    prompt_id_for_task,
    read_video,
)
from train_wan22_ar_tf_ddp import ARTeacherForcingDense4LoRAModule, _load_ar_checkpoint
from wan22_dense4_checkpoint import build_provenance_metadata, checkpoint_metadata
from wan22_task_router import TASK_TO_ID


THIS_DIR = Path(__file__).resolve().parent
TASKS = ("depth", "normal", "matte", "raymap")
RAW_VIDEO_FRAMES = 81
LATENT_FRAMES = 21
LATENT_SHAPE = (48, 21, 30, 52)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--code-patch", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--depth-cache-dir", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs-per-task", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup_runs < 1 or args.runs_per_task < 1:
        raise ValueError("Benchmark run counts must be positive")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    metadata = checkpoint_metadata(args.checkpoint)
    prompt_texts = json.loads(metadata["output_prompt_texts"])
    prompt_ids = {
        task: prompt_id_for_task(prompt_texts, task, "person") for task in TASKS
    }
    module = ARTeacherForcingDense4LoRAModule(
        str(args.model_dir.resolve()),
        int(metadata["lora_rank"]),
        metadata["lora_targets"],
        int(metadata["router_hidden_dim"]),
        metadata["router_mode"],
        0.0,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(prompt_texts, device)
    provenance = build_provenance_metadata(
        module,
        model_dir=args.model_dir,
        prompt_texts=prompt_texts,
        prompt_dir=args.prompt_dir,
        depth_cache_dir=args.depth_cache_dir,
        target_cache_dir=args.target_cache_dir,
        code_patch_path=args.code_patch,
        repo_dir=THIS_DIR.parents[3],
    )
    provenance["diffsynth_git_head"] = metadata["diffsynth_git_head"]
    module.pipe.dit.to(device).eval()
    _load_ar_checkpoint(
        module,
        args.checkpoint,
        expected_provenance=provenance,
        lora_rank=int(metadata["lora_rank"]),
        lora_targets=metadata["lora_targets"],
        router_hidden_dim=int(metadata["router_hidden_dim"]),
        expected_router_mode=metadata["router_mode"],
    )
    vae = load_official_vae(args.official_repo.resolve(), args.model_dir.resolve(), device)
    input_frames, _ = read_video(args.source_video.resolve())

    generator = torch.Generator(device="cpu").manual_seed(20260904)
    source = torch.randn((1, *LATENT_SHAPE), generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    )

    def run(task: str) -> float:
        task_id = torch.tensor([TASK_TO_ID[task]], device=device, dtype=torch.long)
        prompt_id = torch.tensor([prompt_ids[task]], device=device, dtype=torch.long)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output, _ = module.predict_free_running(source, task_id, prompt_id)
                if tuple(output.shape) != tuple(source.shape) or not torch.isfinite(output).all():
                    raise RuntimeError("Invalid benchmark output")
        finally:
            module.task_router.clear()
        torch.cuda.synchronize()
        return time.perf_counter() - started

    for _ in range(args.warmup_runs):
        run("depth")
    measurements = {task: [run(task) for _ in range(args.runs_per_task)] for task in TASKS}
    all_seconds = [value for values in measurements.values() for value in values]

    def run_end_to_end() -> float:
        torch.cuda.synchronize()
        started = time.perf_counter()
        source_cpu = encode_video(vae, input_frames, device)
        task_id = torch.tensor([TASK_TO_ID["depth"]], device=device, dtype=torch.long)
        prompt_id = torch.tensor([prompt_ids["depth"]], device=device, dtype=torch.long)
        try:
            with torch.inference_mode():
                prediction, _ = module.predict_free_running(
                    source_cpu.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                    task_id,
                    prompt_id,
                )
        finally:
            module.task_router.clear()
        decoded = decode_video(vae, prediction[0], device)
        if tuple(decoded.shape) != (3, 81, 480, 832):
            raise RuntimeError("Invalid end-to-end benchmark output")
        torch.cuda.synchronize()
        return time.perf_counter() - started

    run_end_to_end()
    end_to_end_seconds = [run_end_to_end() for _ in range(3)]

    def stats(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        mean = statistics.mean(values)
        p50 = statistics.median(values)
        p90 = ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]
        return {
            "mean_seconds_per_81_frame_clip": mean,
            "p50_seconds_per_81_frame_clip": p50,
            "p90_seconds_per_81_frame_clip": p90,
            "raw_video_fps_mean": RAW_VIDEO_FRAMES / mean,
            "latent_fps_mean": LATENT_FRAMES / mean,
        }

    report = {
        "schema": "wan22_ar_free_running_core_benchmark_v1",
        "scope": "DiT free-running AR only; excludes model load, VAE encode/decode, and file I/O",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(metadata["step"]),
        "frames_per_chunk": 3,
        "chunks": 7,
        "warmup_runs": args.warmup_runs,
        "runs_per_task": args.runs_per_task,
        "per_task_seconds": measurements,
        "per_task": {task: stats(values) for task, values in measurements.items()},
        "overall": stats(all_seconds),
        "resident_end_to_end_no_file_io": {
            **stats(end_to_end_seconds),
            "seconds": end_to_end_seconds,
            "scope": "RGB tensor -> VAE encode -> free-running AR -> VAE decode -> CPU tensor",
        },
        "gpu": torch.cuda.get_device_name(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
