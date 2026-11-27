"""Evaluate the final Wan2.1 Dense4 checkpoint on four fixed training clips.

One torchrun rank owns one clip and evaluates all four routed tasks.  Ground
truth and RGB are decoded from the exact BF16 cache used during training, so
the comparison isolates learned latent regression from raw-data reader drift.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw

from dense4_checkpoint import load_dense4_checkpoint
from train_dense4_ddp import CachedDense4Dataset, Dense4LoRAModule, ID_TO_TASK, TASK_TO_ID
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


MODEL_DIR = Path("/pfs/mayuema/dense/wan/Wan2.1-T2V-1.3B")
DEPTH_CACHE = Path("/pfs/mayuema/dense/wan/cache/depth_latents_bf16_bright40")
TARGET_CACHE = Path("/pfs/mayuema/dense/wan/cache/dense4_targets_bf16_bright40")
CHECKPOINT = Path("/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k/step-10000.safetensors")
OUTPUT_ROOT = Path("/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_train_eval4")

CASES = (
    ("person", "0a1afa764b9fb300/008f8fd4bbb646de9069"),
    ("bear", "0a1afa764b9fb300/0ca34dd9ac5c62dc102f"),
    ("robot", "0a1afa764b9fb300/753a7d17c2140c7cdfd6"),
    ("dog", "25ec11661f25829e/133d7024587e1048be36"),
)


def decoded_video(vae, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return a finite CPU float tensor in C,T,H,W layout."""
    value = vae.decode(latent.unsqueeze(0), device=device, tiled=False)[0]
    value = value.detach().float().cpu().contiguous()
    if tuple(value.shape) != (3, 81, 480, 832) or not torch.isfinite(value).all():
        raise ValueError(f"Invalid decoded video: {tuple(value.shape)}")
    return value


def rgb_uint8(video: torch.Tensor, grayscale: bool = False) -> np.ndarray:
    value = video.clamp(-1, 1)
    if grayscale:
        value = value.mean(dim=0, keepdim=True).repeat(3, 1, 1, 1)
    value = ((value + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return value.permute(1, 2, 3, 0).numpy()


def error_uint8(prediction: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    error = (prediction - target).abs().mean(dim=0).clamp(0, 2) / 2.0
    # Black-to-yellow error map: high error is visually obvious without an
    # external colormap dependency.
    red = (error * 255).round().to(torch.uint8)
    green = (error.sqrt() * 180).round().clamp(0, 255).to(torch.uint8)
    blue = torch.zeros_like(red)
    return torch.stack((red, green, blue), dim=-1).numpy()


def label_panel(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 330), 27), fill=(0, 0, 0))
    draw.text((7, 6), text, fill=(255, 255, 255))
    return np.asarray(image)


def save_comparison(
    path: Path,
    source: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    task: str,
    clip_id: str,
) -> None:
    grayscale = task in ("depth", "matte")
    source_frames = rgb_uint8(source)
    prediction_frames = rgb_uint8(prediction, grayscale=grayscale)
    target_frames = rgb_uint8(target, grayscale=grayscale)
    errors = error_uint8(prediction, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=24, quality=7, macro_block_size=16)
    selected = {0, 20, 40, 60, 80}
    preview_rows = []
    try:
        for index in range(81):
            panels = (
                label_panel(source_frames[index], "RGB input"),
                label_panel(prediction_frames[index], f"prediction: {task}"),
                label_panel(target_frames[index], "VAE-decoded train GT"),
                label_panel(errors[index], "absolute error"),
            )
            combined = np.concatenate(panels, axis=1)
            writer.append_data(combined)
            if index in selected:
                preview_rows.append(combined)
    finally:
        writer.close()
    Image.fromarray(np.concatenate(preview_rows, axis=0)).save(path.with_name("preview.png"))
    path.with_name("case.txt").write_text(f"clip_id={clip_id}\ntask={task}\n", encoding="utf-8")


def metrics(
    pred_latent: torch.Tensor,
    target_latent: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    task: str,
) -> dict[str, float]:
    latent_delta = pred_latent.float() - target_latent.float()
    pixel_delta = prediction - target
    latent_mse = float(latent_delta.square().mean())
    pixel_mse = float(pixel_delta.square().mean())
    result = {
        "latent_mae": float(latent_delta.abs().mean()),
        "latent_mse": latent_mse,
        "latent_cosine": float(torch.nn.functional.cosine_similarity(
            pred_latent.float().flatten(), target_latent.float().flatten(), dim=0
        )),
        "pixel_mae": float(pixel_delta.abs().mean()),
        "pixel_mse": pixel_mse,
        "pixel_psnr_db": float(10.0 * math.log10(4.0 / max(pixel_mse, 1.0e-12))),
    }
    if task == "matte":
        pred_mask = prediction.mean(dim=0) > 0
        target_mask = target.mean(dim=0) > 0
        intersection = float((pred_mask & target_mask).sum())
        union = float((pred_mask | target_mask).sum())
        denom = float(pred_mask.sum() + target_mask.sum())
        result["matte_iou"] = intersection / max(union, 1.0)
        result["matte_f1"] = 2.0 * intersection / max(denom, 1.0)
    if task in ("normal", "raymap"):
        pred_vec = torch.nn.functional.normalize(prediction, dim=0, eps=1.0e-6)
        target_vec = torch.nn.functional.normalize(target, dim=0, eps=1.0e-6)
        cosine = (pred_vec * target_vec).sum(dim=0).clamp(-1, 1)
        result[f"{task}_angular_deg"] = float(torch.rad2deg(torch.acos(cosine)).mean())
    return result


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != len(CASES):
        raise ValueError(f"Expected {len(CASES)} ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo")
    torch.manual_seed(20260830)

    subject, clip_id = CASES[rank]
    dataset = CachedDense4Dataset(DEPTH_CACHE, TARGET_CACHE)
    try:
        clip_index = dataset.clip_ids.index(clip_id)
    except ValueError as error:
        raise ValueError(f"Selected clip is absent from training cache: {clip_id}") from error

    module = Dense4LoRAModule(
        str(MODEL_DIR),
        lora_rank=32,
        target_modules="q,k,v,o,ffn.0,ffn.2",
        router_hidden_dim=512,
        router_mode="learned",
        router_identity_weight=1.0e-3,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(dataset.prompt_texts, device)
    module.pipe.dit.to(device).eval()
    metadata = load_dense4_checkpoint(
        module,
        CHECKPOINT,
        model_dir=MODEL_DIR,
        lora_rank=32,
        lora_targets="q,k,v,o,ffn.0,ffn.2",
        router_hidden_dim=512,
        expected_router_mode="learned",
    )

    vae_pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=[ModelConfig(path=str(MODEL_DIR / "Wan2.1_VAE.pth"))],
        tokenizer_config=None,
    )
    if vae_pipe.vae is None:
        raise RuntimeError("Wan2.1 VAE did not load")
    vae_pipe.vae.to(device).eval()

    source_latent = dataset[clip_index * len(TASK_TO_ID)][0]
    source_video = decoded_video(vae_pipe.vae, source_latent, device)
    rows = []
    with torch.inference_mode():
        for task_id in range(len(TASK_TO_ID)):
            task = ID_TO_TASK[task_id]
            source, target, cached_task_id, prompt_id = dataset[
                clip_index * len(TASK_TO_ID) + task_id
            ]
            if cached_task_id != task_id or not torch.equal(source, source_latent):
                raise RuntimeError(f"Cache routing mismatch for {clip_id}/{task}")
            velocity, _ = module.predict_velocity(
                source.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                torch.tensor([task_id], device=device, dtype=torch.long),
                torch.tensor([prompt_id], device=device, dtype=torch.long),
            )
            pred_latent = (-velocity[0]).detach().to(device="cpu", dtype=torch.bfloat16)
            prediction = decoded_video(vae_pipe.vae, pred_latent, device)
            target_video = decoded_video(vae_pipe.vae, target, device)
            row = {
                "rank": rank,
                "subject": subject,
                "clip_id": clip_id,
                "task": task,
                "prompt": dataset.prompt_texts[prompt_id],
                **metrics(pred_latent, target, prediction, target_video, task),
            }
            rows.append(row)
            task_dir = OUTPUT_ROOT / subject / task
            save_comparison(
                task_dir / "comparison.mp4",
                source_video,
                prediction,
                target_video,
                task,
                clip_id,
            )
            (task_dir / "metrics.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del prediction, target_video, pred_latent, velocity
            torch.cuda.empty_cache()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / f"metrics_rank{rank}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dist.barrier()
    if rank == 0:
        all_rows = []
        for worker_rank in range(world_size):
            all_rows.extend(json.loads((OUTPUT_ROOT / f"metrics_rank{worker_rank}.json").read_text()))
        fieldnames = list(all_rows[0])
        for row in all_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (OUTPUT_ROOT / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        aggregate = {}
        for task in ID_TO_TASK.values():
            task_rows = [row for row in all_rows if row["task"] == task]
            numeric_keys = sorted(
                key for key in set().union(*(row.keys() for row in task_rows))
                if all(isinstance(row.get(key), (int, float)) for row in task_rows)
                and key != "rank"
            )
            aggregate[task] = {
                key: float(np.mean([row[key] for row in task_rows])) for key in numeric_keys
            }
        summary = {
            "schema": "wan21_dense4_train_eval4_v1",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_step": int(metadata["step"]),
            "ground_truth": "VAE-decoded BF16 training target cache",
            "cases": [{"subject": subject, "clip_id": clip_id} for subject, clip_id in CASES],
            "aggregate": aggregate,
        }
        (OUTPUT_ROOT / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (OUTPUT_ROOT / "EVAL_COMPLETE").write_text(
            json.dumps({"rows": len(all_rows), "checkpoint_step": int(metadata["step"])}),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
