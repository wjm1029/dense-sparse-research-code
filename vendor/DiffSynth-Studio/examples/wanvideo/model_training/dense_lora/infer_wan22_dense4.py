"""Single-forward Wan2.2 Dense4 inference with the trained routed LoRA.

The model consumes a clean Wan2.2 latent at t=0 and predicts the requested
dense target in one DiT forward.  Depth, normal, alpha matte and camera raymap
share one LoRA; the checkpointed task HyperNet selects the route.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_wan22_dense4_ddp import Dense4LoRAModule  # noqa: E402
from wan22_cache_integrity import sha256_file  # noqa: E402
from wan22_dense4_checkpoint import (  # noqa: E402
    build_provenance_metadata,
    checkpoint_metadata,
    load_dense4_checkpoint,
)
from wan22_task_router import TASK_TO_ID  # noqa: E402


TASKS = tuple(TASK_TO_ID)
LATENT_SHAPE = (48, 21, 30, 52)
VIDEO_SHAPE = (81, 480, 832, 3)
DEFAULT_ROOT = Path("/pfs/mayuema/dense/wan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-forward Wan2.2 Dense4 routed-LoRA inference."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_video", help="81-frame 832x480 RGB video")
    source.add_argument(
        "--source_latent",
        help=".pt tensor or payload containing source_latent with shape [48,21,30,52]",
    )
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument(
        "--tasks",
        default="all",
        help="all or comma-separated subset of depth,normal,matte,raymap",
    )
    parser.add_argument(
        "--matte_subject",
        default="person",
        help="subject label used by the matte output prompt (default: person)",
    )
    parser.add_argument("--fps", type=float, default=0.0, help="override output fps")
    parser.add_argument("--min_rgb_mean", type=float, default=40.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_tasks(value: str) -> list[str]:
    tasks = list(TASKS) if value.strip().lower() == "all" else [x.strip() for x in value.split(",")]
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASK_TO_ID for task in tasks):
        raise ValueError(f"--tasks must be all or a unique subset of {','.join(TASKS)}")
    return tasks


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if existing and not overwrite:
        raise FileExistsError(f"Output directory is not empty (pass --overwrite): {path}")
    if overwrite:
        for name in ("INFERENCE_COMPLETE", "manifest.json"):
            candidate = path / name
            if candidate.exists():
                candidate.unlink()


def read_video(path: Path) -> tuple[np.ndarray, float]:
    reader = imageio.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        frames = [np.asarray(frame)[..., :3] for frame in reader]
    finally:
        reader.close()
    video = np.stack(frames).astype(np.uint8, copy=False)
    if tuple(video.shape) != VIDEO_SHAPE:
        raise ValueError(f"Expected video shape {VIDEO_SHAPE}, got {tuple(video.shape)}: {path}")
    fps = float(metadata.get("fps") or 16.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 16.0
    return video, fps


def load_source_latent(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload.get("source_latent") if isinstance(payload, dict) else payload
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"No source_latent tensor in {path}")
    value = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(value.shape) != LATENT_SHAPE or not torch.isfinite(value).all():
        raise ValueError(f"Invalid source latent {tuple(value.shape)}: {path}")
    return value


def load_official_vae(official_repo: Path, model_dir: Path, device: torch.device):
    source = official_repo / "wan/modules/vae2_2.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing official Wan2.2 VAE source: {source}")
    checkpoint = model_dir / "Wan2.2_VAE.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Wan2.2 VAE checkpoint: {checkpoint}")
    spec = importlib.util.spec_from_file_location("wan22_official_vae_inference", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official Wan2.2 VAE: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Wan2_2_VAE(
        vae_pth=str(checkpoint), dtype=torch.bfloat16, device=str(device)
    )


@torch.no_grad()
def encode_video(vae, frames: np.ndarray, device: torch.device) -> torch.Tensor:
    video = torch.from_numpy(frames.copy()).float().div_(127.5).sub_(1.0)
    video = video.permute(3, 0, 1, 2).contiguous().to(device=device, dtype=torch.bfloat16)
    encoded = vae.encode([video])
    if not encoded or encoded[0] is None:
        raise RuntimeError("Official Wan2.2 VAE encode returned no latent")
    latent = encoded[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(latent.shape) != LATENT_SHAPE or not torch.isfinite(latent).all():
        raise ValueError(f"Invalid encoded latent: {tuple(latent.shape)}")
    return latent


@torch.no_grad()
def decode_video(vae, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    decoded = vae.decode([latent.to(device=device, dtype=torch.bfloat16)])
    if not decoded or decoded[0] is None:
        raise RuntimeError("Official Wan2.2 VAE decode returned no video")
    value = decoded[0].detach().float().cpu().contiguous()
    expected = (3, 81, 480, 832)
    if tuple(value.shape) != expected or not torch.isfinite(value).all():
        raise ValueError(f"Invalid decoded video: {tuple(value.shape)}")
    return value


def rgb_frames(video: torch.Tensor) -> np.ndarray:
    value = ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return value.permute(1, 2, 3, 0).numpy()


def task_frames(video: torch.Tensor, task: str) -> np.ndarray:
    if task in ("depth", "matte"):
        scalar = video.mean(dim=0).clamp(-1, 1).add(1.0).mul(127.5).round().to(torch.uint8)
        return np.repeat(scalar.numpy()[..., None], 3, axis=-1)
    return rgb_frames(video)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 440), 28), fill=(0, 0, 0))
    draw.text((7, 7), text, fill=(255, 255, 255))
    return np.asarray(image)


def write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, quality=7, macro_block_size=16)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def write_task_outputs(
    output_dir: Path,
    task: str,
    source: np.ndarray,
    prediction: torch.Tensor,
    latent: torch.Tensor,
    fps: float,
) -> None:
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    predicted = task_frames(prediction, task)
    comparisons = np.stack(
        [
            np.concatenate(
                (label(source[i], "RGB input"), label(predicted[i], f"prediction: {task}")), axis=1
            )
            for i in range(VIDEO_SHAPE[0])
        ]
    )
    write_video(task_dir / "prediction.mp4", predicted, fps)
    write_video(task_dir / "comparison.mp4", comparisons, fps)
    preview_indices = (0, 20, 40, 60, 80)
    Image.fromarray(np.concatenate([comparisons[i] for i in preview_indices], axis=0)).save(
        task_dir / "preview.png"
    )
    torch.save(
        {"schema": "wan22_dense4_inference_latent_v1", "task": task, "latent": latent},
        task_dir / "prediction_latent.pt",
    )


def prompt_id_for_task(prompt_texts: list[str], task: str, matte_subject: str) -> int:
    prompt = {
        "depth": "output: depth",
        "normal": "output: surface normal",
        "raymap": "output: camera raymap",
        "matte": f"output: {matte_subject.strip().lower()} alpha matte",
    }[task]
    try:
        return prompt_texts.index(prompt)
    except ValueError as error:
        matte_prompts = [text for text in prompt_texts if text.endswith(" alpha matte")]
        raise ValueError(
            f"Prompt {prompt!r} is absent from the checkpoint. Available matte prompts: {matte_prompts}"
        ) from error


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Wan2.2 Dense4 inference requires CUDA")
    torch.cuda.set_device(device)
    tasks = selected_tasks(args.tasks)
    output_dir = Path(args.output_dir).resolve()
    prepare_output_dir(output_dir, args.overwrite)

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
    source_path: Path
    if args.input_video:
        source_path = Path(args.input_video).resolve()
        input_frames, detected_fps = read_video(source_path)
        rgb_mean = float(input_frames.mean())
        if rgb_mean < args.min_rgb_mean:
            raise ValueError(
                f"Input RGB mean {rgb_mean:.3f} is below the training-domain threshold "
                f"{args.min_rgb_mean}; use --min_rgb_mean 0 to override"
            )
        source_latent = encode_video(vae, input_frames, device)
    else:
        source_path = Path(args.source_latent).resolve()
        source_latent = load_source_latent(source_path)
        input_frames = rgb_frames(decode_video(vae, source_latent, device))
        detected_fps = 16.0
        rgb_mean = float(input_frames.mean())
    fps = args.fps if args.fps > 0 else detected_fps

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
                output_dir, task, input_frames, prediction, prediction_latent, fps
            )
            row = {
                "task": task,
                "task_id": task_id,
                "prompt": prompt_texts[prompt_ids[task]],
                "gate_min": float(gates.min()),
                "gate_max": float(gates.max()),
                "gate_mean": float(gates.mean()),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del velocity, gates, prediction_latent, prediction
            torch.cuda.empty_cache()

    manifest = {
        "schema": "wan22_dense4_single_forward_inference_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(loaded_metadata["step"]),
        "base_model": str(model_dir),
        "vae_checkpoint": str(model_dir / "Wan2.2_VAE.pth"),
        "source": str(source_path),
        "source_kind": "video" if args.input_video else "latent",
        "source_rgb_mean": rgb_mean,
        "source_latent_shape": list(source_latent.shape),
        "fps": fps,
        "tasks": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "INFERENCE_COMPLETE").write_text(
        json.dumps({"checkpoint_step": int(loaded_metadata["step"]), "tasks": tasks}),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
