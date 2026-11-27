"""Build the full GenCeption-style depth latent cache with one process per GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import DepthVideoDataset  # noqa: E402
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline  # noqa: E402


CACHE_SCHEMA = "wan21_depth_latent_v1"
LATENT_SHAPE = (16, 21, 60, 104)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--min_rgb_mean", type=float, default=40.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--vae_tiled", action="store_true")
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    return rank, local_rank, world_size, device


def cache_path(cache_dir: Path, clip_id: str) -> Path:
    scene_id, clip_name = clip_id.split("/", 1)
    return cache_dir / scene_id / f"{clip_name}.pt"


def existing_cache_is_valid(path: Path) -> bool:
    # A valid pair is roughly 8 MiB in BF16. Avoid rereading every tensor when
    # resuming a large cache job, but reject empty/truncated files.
    return path.is_file() and path.stat().st_size > 4 * 1024 * 1024


def load_or_build_dataset(
    args: argparse.Namespace,
    cache_dir: Path,
    rank: int,
    world_size: int,
) -> DepthVideoDataset:
    """Scan GPFS once on rank 0, then let every rank load the compact index."""
    index_path = cache_dir / "dataset_index.json"
    if rank == 0 and not index_path.is_file():
        dataset = DepthVideoDataset(
            args.dataset_root,
            min_rgb_mean=args.min_rgb_mean,
            max_samples=args.max_samples,
        )
        temporary = index_path.with_suffix(f".tmp-p{os.getpid()}")
        temporary.write_text(
            json.dumps(dataset.index_payload(), separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, index_path)
        print(f"Wrote shared dataset index: {index_path}", flush=True)
    if world_size > 1:
        dist.barrier()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    dataset = DepthVideoDataset.from_index_payload(
        payload,
        root=args.dataset_root,
        min_rgb_mean=args.min_rgb_mean,
    )
    if args.max_samples > 0 and len(dataset) != args.max_samples:
        raise ValueError(
            f"Shared dataset index has {len(dataset)} samples, expected max_samples={args.max_samples}"
        )
    return dataset


@torch.no_grad()
def encode_video(pipe: WanVideoPipeline, video_tchw: torch.Tensor, device: torch.device, tiled: bool) -> torch.Tensor:
    video_cthw = video_tchw.permute(1, 0, 2, 3).contiguous().to(torch.bfloat16)
    latent = pipe.vae.encode(
        video_cthw.unsqueeze(0),
        device=device,
        tiled=tiled,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    )[0]
    latent = latent.to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(latent.shape) != LATENT_SHAPE or not torch.isfinite(latent).all():
        raise ValueError(f"Invalid VAE latent: shape={tuple(latent.shape)}, finite={torch.isfinite(latent).all()}")
    return latent


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = init_distributed()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_or_build_dataset(args, cache_dir, rank, world_size)
    if rank == 0:
        manifest = {
            "schema": CACHE_SCHEMA,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "model_dir": str(Path(args.model_dir).resolve()),
            "min_rgb_mean": args.min_rgb_mean,
            "num_samples": len(dataset),
            "scan_stats": dataset.scan_stats,
            "latent_shape": list(LATENT_SHAPE),
            "latent_dtype": "bfloat16",
            "prompt": dataset.prompt,
        }
        (cache_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2), flush=True)

    model_dir = Path(args.model_dir).resolve()
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=[ModelConfig(path=str(model_dir / "Wan2.1_VAE.pth"))],
        tokenizer_config=None,
    )
    if pipe.vae is None:
        raise RuntimeError("Wan VAE did not load")
    pipe.device = device
    pipe.vae.to(device).eval()

    indices = range(rank, len(dataset), world_size)
    written = 0
    skipped = 0
    progress = tqdm(indices, desc=f"cache rank {rank}", disable=local_rank != 0)
    for index in progress:
        row = dataset.samples[index]
        destination = cache_path(cache_dir, row["id"])
        if existing_cache_is_valid(destination):
            skipped += 1
            continue
        sample = dataset[index]
        source_latent = encode_video(pipe, sample["source_video"], device, args.vae_tiled)
        target_latent = encode_video(pipe, sample["target_video"], device, args.vae_tiled)
        payload = {
            "schema": CACHE_SCHEMA,
            "clip_id": sample["clip_id"],
            "prompt": sample["prompt"],
            "rgb_mean": float(sample["rgb_mean"]),
            "depth_p2": float(sample["depth_p2"]),
            "depth_p98": float(sample["depth_p98"]),
            "source_latent": source_latent,
            "target_latent": target_latent,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".tmp-r{rank}-p{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        written += 1
        del sample, source_latent, target_latent, payload

    print(f"rank={rank} written={written} skipped={skipped}", flush=True)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        files = sorted(cache_dir.glob("*/*.pt"))
        if len(files) != len(dataset):
            raise RuntimeError(f"Cache incomplete: expected {len(dataset)}, found {len(files)}")
        (cache_dir / "CACHE_COMPLETE").write_text(
            json.dumps({"schema": CACHE_SCHEMA, "num_samples": len(files)}, indent=2),
            encoding="utf-8",
        )
        print(f"CACHE COMPLETE: {len(files)} samples", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
