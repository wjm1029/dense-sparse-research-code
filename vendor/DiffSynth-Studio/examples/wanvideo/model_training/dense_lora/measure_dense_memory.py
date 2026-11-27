"""Measure real single-GPU peak memory for Dense4 full tuning and inference."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


ROOT = Path("/pfs/mayuema/dense/wan")
VIDEO = Path("/pfs/mayuema/dense/ditto/videos_832x480_16fps_81f/5495888_pavel_danilyuk_1.mp4")


def gib(value: int) -> float:
    return value / 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("wan21", "wan22"), required=True)
    parser.add_argument("--mode", choices=("full_train", "inference"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build(args: argparse.Namespace, device: torch.device):
    if args.version == "wan21":
        from train_dense4_ddp import CachedDense4Dataset, Dense4LoRAModule

        dataset = CachedDense4Dataset(
            ROOT / "cache/depth_latents_bf16_bright40",
            ROOT / "cache/dense4_targets_bf16_bright40",
        )
        module = Dense4LoRAModule(
            str(ROOT / "Wan2.1-T2V-1.3B"), 32, "q,k,v,o,ffn.0,ffn.2", 512, "learned", 1.0e-3
        )
    else:
        from train_wan22_dense4_ddp import CachedDense4Dataset, Dense4LoRAModule

        dataset = CachedDense4Dataset(
            ROOT / "cache/wan22_depth_latents_bf16_bright40",
            ROOT / "cache/wan22_dense4_targets_bf16_bright40",
            allow_legacy_v1_cache=True,
        )
        module = Dense4LoRAModule(
            str(ROOT / "Wan2.2-TI2V-5B"), 32, "q,k,v,o,ffn.0,ffn.2", 512, "learned", 1.0e-3
        )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(dataset.prompt_texts, device)
    module.pipe.dit.to(device)
    return dataset, module


def measure_full_train(args: argparse.Namespace, device: torch.device, dataset, module) -> dict:
    module.pipe.dit.train()
    for parameter in module.pipe.dit.parameters():
        parameter.requires_grad_(True)
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    total_dit = sum(parameter.numel() for parameter in module.pipe.dit.parameters())
    trainable = sum(parameter.numel() for parameter in parameters)
    dtype_counts: dict[str, int] = {}
    for parameter in parameters:
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-4)
    source, target, task_id, prompt_id = dataset[0]
    source = source.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    target = target.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    task_ids = torch.tensor([task_id], device=device, dtype=torch.long)
    prompt_ids = torch.tensor([prompt_id], device=device, dtype=torch.long)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    loss, data_loss, identity_loss = module(
        source, target, task_ids, prompt_ids, gradient_checkpointing=True
    )
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    state_dtype_counts: dict[str, int] = {}
    state_elements = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                state_elements += value.numel()
                key = str(value.dtype)
                state_dtype_counts[key] = state_dtype_counts.get(key, 0) + value.numel()
    return {
        "loss": float(loss.detach()),
        "total_dit_parameters": total_dit,
        "trainable_parameters": trainable,
        "trainable_dtype_counts": dtype_counts,
        "optimizer_state_elements": state_elements,
        "optimizer_state_dtype_counts": state_dtype_counts,
        "baseline_allocated_gib": gib(baseline_allocated),
        "baseline_reserved_gib": gib(baseline_reserved),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
    }


def measure_inference(args: argparse.Namespace, device: torch.device, dataset, module) -> dict:
    module.pipe.dit.eval()
    if args.version == "wan21":
        from infer_wan21_dense4_ditto20 import decode_video, encode_video, read_video

        frames = read_video(VIDEO)
        vae_pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[ModelConfig(path=str(ROOT / "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))],
            tokenizer_config=None,
        )
        vae = vae_pipe.vae
        if vae is None:
            raise RuntimeError("Wan2.1 VAE failed to load")
        vae.to(device).eval()
    else:
        from infer_wan22_dense4 import decode_video, encode_video, load_official_vae, read_video

        frames, _ = read_video(VIDEO)
        vae = load_official_vae(ROOT / "Wan2.2", ROOT / "Wan2.2-TI2V-5B", device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    source_latent = encode_video(vae, frames, device)
    with torch.inference_mode():
        for task_id, prompt_id in ((0, 0), (1, 1)):
            velocity, _ = module.predict_velocity(
                source_latent.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                torch.tensor([task_id], device=device, dtype=torch.long),
                torch.tensor([prompt_id], device=device, dtype=torch.long),
            )
            prediction_latent = (-velocity[0]).detach().cpu().to(torch.bfloat16)
            prediction = decode_video(vae, prediction_latent, device)
            del velocity, prediction_latent, prediction
            torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return {
        "tasks": ["depth", "normal"],
        "baseline_allocated_gib": gib(baseline_allocated),
        "baseline_reserved_gib": gib(baseline_reserved),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    result = {"version": args.version, "mode": args.mode, "status": "starting"}
    try:
        dataset, module = build(args, device)
        if args.mode == "full_train":
            result.update(measure_full_train(args, device, dataset, module))
        else:
            result.update(measure_inference(args, device, dataset, module))
        result["status"] = "ok"
    except torch.OutOfMemoryError as error:
        result.update(
            {
                "status": "oom",
                "error": str(error),
                "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
            }
        )
    except Exception as error:
        result.update({"status": "error", "error": repr(error), "traceback": traceback.format_exc()})
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
