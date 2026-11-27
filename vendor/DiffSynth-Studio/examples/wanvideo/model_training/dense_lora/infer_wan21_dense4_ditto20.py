"""Run the final Wan2.1 Dense4 checkpoint on 20 brightness-stratified Ditto videos."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dense4_checkpoint import load_dense4_checkpoint  # noqa: E402
from train_dense4_ddp import Dense4LoRAModule, TASK_TO_ID  # noqa: E402
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402


INPUT_ROOT = Path("/pfs/mayuema/dense/ditto/videos_832x480_16fps_81f")
MODEL_DIR = Path("/pfs/mayuema/dense/wan/Wan2.1-T2V-1.3B")
CHECKPOINT = Path("/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k/step-10000.safetensors")
OUTPUT_ROOT = Path("/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_ditto20_depth_normal")
PROMPTS = ("output: depth", "output: surface normal")
FRAME_INDICES = (0, 20, 40, 60, 80)
LATENT_SHAPE = (16, 21, 60, 104)


def read_video(path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(path))
    try:
        frames = [np.asarray(frame)[..., :3] for frame in reader]
    finally:
        reader.close()
    video = np.stack(frames).astype(np.uint8, copy=False)
    if video.shape != (81, 480, 832, 3):
        raise ValueError(f"Unexpected video shape {video.shape}: {path}")
    return video


def brightness(path: Path) -> float:
    reader = imageio.get_reader(str(path))
    try:
        values = [float(np.asarray(reader.get_data(index))[..., :3].mean()) for index in FRAME_INDICES]
    finally:
        reader.close()
    return float(np.mean(values))


def select_cases() -> list[dict]:
    paths = sorted(INPUT_ROOT.glob("*.mp4"))
    if len(paths) < 20:
        raise ValueError(f"Expected at least 20 mp4 files, found {len(paths)}")
    ranked = sorted(((brightness(path), path) for path in paths), key=lambda item: (item[0], item[1].name))
    positions = np.linspace(0, len(ranked) - 1, 20).round().astype(int).tolist()
    if len(set(positions)) != 20:
        raise RuntimeError(f"Brightness selection produced duplicates: {positions}")
    return [
        {"case_index": index, "case_name": path.stem, "path": str(path), "brightness": value}
        for index, (value, path) in enumerate(ranked[position] for position in positions)
    ]


@torch.no_grad()
def encode_video(vae, frames: np.ndarray, device: torch.device) -> torch.Tensor:
    video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    video = video.permute(1, 0, 2, 3).contiguous().to(torch.bfloat16)
    latent = vae.encode(video.unsqueeze(0), device=device, tiled=False)[0]
    latent = latent.to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(latent.shape) != LATENT_SHAPE or not torch.isfinite(latent).all():
        raise ValueError(f"Invalid source latent: {tuple(latent.shape)}")
    return latent


@torch.no_grad()
def decode_video(vae, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = vae.decode(latent.unsqueeze(0), device=device, tiled=False)[0]
    value = value.detach().float().cpu().contiguous()
    if tuple(value.shape) != (3, 81, 480, 832) or not torch.isfinite(value).all():
        raise ValueError(f"Invalid decoded video: {tuple(value.shape)}")
    return value


def label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 420), 27), fill=(0, 0, 0))
    draw.text((7, 6), text, fill=(255, 255, 255))
    return np.asarray(image)


def write_video(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=16, quality=7, macro_block_size=16)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def prediction_rgb(decoded: torch.Tensor) -> np.ndarray:
    value = ((decoded.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return value.permute(1, 2, 3, 0).numpy()


def depth_views(decoded: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    depth01 = decoded.mean(dim=0).add(1.0).mul(0.5).clamp(0, 1)
    gray = depth01.mul(255).round().to(torch.uint8).numpy()
    gray_rgb = np.repeat(gray[..., None], 3, axis=-1)
    color = np.stack(
        [cv2.cvtColor(cv2.applyColorMap(frame, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB) for frame in gray]
    )
    return gray_rgb, color


def save_preview(path: Path, rows: list[np.ndarray]) -> None:
    Image.fromarray(np.concatenate([rows[index] for index in FRAME_INDICES], axis=0)).save(path)


def save_case(case: dict, source: np.ndarray, task: str, decoded: torch.Tensor) -> None:
    task_dir = OUTPUT_ROOT / case["case_name"] / task
    task_dir.mkdir(parents=True, exist_ok=True)
    if task == "depth":
        gray, color = depth_views(decoded)
        comparisons = np.stack(
            [
                np.concatenate(
                    (label(source[i], "RGB input"), label(gray[i], "inverse depth (near=white)"), label(color[i], "inverse depth (Turbo)")),
                    axis=1,
                )
                for i in range(81)
            ]
        )
        write_video(task_dir / "prediction_gray.mp4", gray)
        write_video(task_dir / "prediction_color.mp4", color)
    else:
        prediction = prediction_rgb(decoded)
        comparisons = np.stack(
            [np.concatenate((label(source[i], "RGB input"), label(prediction[i], "surface normal")), axis=1) for i in range(81)]
        )
        write_video(task_dir / "prediction.mp4", prediction)
    write_video(task_dir / "comparison.mp4", comparisons)
    save_preview(task_dir / "preview.png", list(comparisons))
    (task_dir / "metadata.json").write_text(
        json.dumps({**case, "task": task, "checkpoint": str(CHECKPOINT)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_depth_overview(cases: list[dict]) -> None:
    readers = [imageio.get_reader(str(OUTPUT_ROOT / case["case_name"] / "depth" / "prediction_color.mp4")) for case in cases]
    writer = imageio.get_writer(str(OUTPUT_ROOT / "depth_overview_20cases.mp4"), fps=16, quality=7, macro_block_size=16)
    preview = None
    try:
        for frame_index in range(81):
            tiles = []
            for case, reader in zip(cases, readers):
                frame = cv2.resize(np.asarray(reader.get_data(frame_index)), (320, 180), interpolation=cv2.INTER_AREA)
                tiles.append(label(frame, f"{case['case_index']:02d} {case['case_name'][:27]}"))
            grid = np.concatenate([np.concatenate(tiles[row * 5:(row + 1) * 5], axis=1) for row in range(4)], axis=0)
            writer.append_data(grid)
            if frame_index == 40:
                preview = grid
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    if preview is None:
        raise RuntimeError("Depth overview did not produce its preview frame")
    Image.fromarray(preview).save(OUTPUT_ROOT / "depth_overview_frame41.png")


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise ValueError(f"Expected 4 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo")
    torch.manual_seed(20260830)

    cases = select_cases()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (OUTPUT_ROOT / "selection_manifest.json").write_text(
            json.dumps({"selection": "20 brightness quantiles from 30 videos, using frames 1/21/41/61/81", "cases": cases}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    module = Dense4LoRAModule(
        str(MODEL_DIR), lora_rank=32, target_modules="q,k,v,o,ffn.0,ffn.2",
        router_hidden_dim=512, router_mode="learned", router_identity_weight=1.0e-3,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(list(PROMPTS), device)
    module.pipe.dit.to(device).eval()
    metadata = load_dense4_checkpoint(
        module, CHECKPOINT, model_dir=MODEL_DIR, lora_rank=32,
        lora_targets="q,k,v,o,ffn.0,ffn.2", router_hidden_dim=512,
        expected_router_mode="learned",
    )
    vae_pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cpu",
        model_configs=[ModelConfig(path=str(MODEL_DIR / "Wan2.1_VAE.pth"))], tokenizer_config=None,
    )
    if vae_pipe.vae is None:
        raise RuntimeError("Wan2.1 VAE did not load")
    vae_pipe.vae.to(device).eval()

    completed = []
    with torch.inference_mode():
        for case in cases[rank::world_size]:
            source_frames = read_video(Path(case["path"]))
            source_latent = encode_video(vae_pipe.vae, source_frames, device)
            for prompt_id, task in enumerate(("depth", "normal")):
                task_id = TASK_TO_ID[task]
                velocity, _ = module.predict_velocity(
                    source_latent.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                    torch.tensor([task_id], device=device, dtype=torch.long),
                    torch.tensor([prompt_id], device=device, dtype=torch.long),
                )
                prediction = decode_video(vae_pipe.vae, (-velocity[0]).detach().cpu().to(torch.bfloat16), device)
                save_case(case, source_frames, task, prediction)
                completed.append({"case_name": case["case_name"], "task": task})
                print(json.dumps(completed[-1]), flush=True)
                del velocity, prediction
                torch.cuda.empty_cache()
            del source_frames, source_latent
            torch.cuda.empty_cache()

    (OUTPUT_ROOT / f"completed_rank{rank}.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    dist.barrier()
    if rank == 0:
        all_completed = []
        for worker_rank in range(world_size):
            all_completed.extend(json.loads((OUTPUT_ROOT / f"completed_rank{worker_rank}.json").read_text()))
        if len(all_completed) != 40:
            raise RuntimeError(f"Expected 40 completed outputs, got {len(all_completed)}")
        build_depth_overview(cases)
        (OUTPUT_ROOT / "INFERENCE_COMPLETE").write_text(
            json.dumps({"checkpoint_step": int(metadata["step"]), "cases": 20, "tasks": ["depth", "normal"]}),
            encoding="utf-8",
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
