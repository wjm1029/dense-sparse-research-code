"""Build the full Wan2.2 RGB/depth latent cache with one process per GPU.

The preprocessing remains identical to the reviewed Wan2.1 dense integration.
Only the encoder and latent contract change: this script loads the official
Wan2.2 ``Wan2_2_VAE`` implementation and its 48-channel TI2V-5B checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import DepthVideoDataset  # noqa: E402
from cache_integrity import sha256_file, sha256_json, sha256_lines  # noqa: E402


CACHE_SCHEMA = "wan22_depth_latent_v2"
LATENT_SHAPE = (48, 21, 30, 52)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--source_dataset_index", required=True)
    parser.add_argument("--official_repo", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--min_rgb_mean", type=float, default=40.0)
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        # Cache workers are embarrassingly parallel. Use Gloo only for the two
        # metadata barriers so this path does not depend on NCCL collectives.
        dist.init_process_group("gloo")
    return rank, local_rank, world_size, device


def cache_path(cache_dir: Path, clip_id: str) -> Path:
    scene_id, clip_name = clip_id.split("/", 1)
    return cache_dir / scene_id / f"{clip_name}.pt"


def _valid_latent(value: object) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == LATENT_SHAPE
        and value.dtype == torch.bfloat16
        and torch.isfinite(value).all().item()
    )


def existing_cache_is_valid(path: Path, *, clip_id: str, prompt: str, contract_hash: str) -> bool:
    """Fully authenticate a resume candidate before skipping its VAE encode."""
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return bool(
            payload.get("schema") == CACHE_SCHEMA
            and payload.get("cache_contract_hash") == contract_hash
            and payload.get("clip_id") == clip_id
            and payload.get("prompt") == prompt
            and math.isfinite(float(payload.get("rgb_mean")))
            and math.isfinite(float(payload.get("depth_p2")))
            and math.isfinite(float(payload.get("depth_p98")))
            and _valid_latent(payload.get("source_latent"))
            and _valid_latent(payload.get("target_latent"))
        )
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def load_or_build_dataset(
    args: argparse.Namespace,
    cache_dir: Path,
    rank: int,
    world_size: int,
) -> DepthVideoDataset:
    """Scan GPFS once on rank 0, then let every rank load the compact index."""
    index_path = cache_dir / "dataset_index.json"
    if rank == 0 and not index_path.is_file():
        source_index = Path(args.source_dataset_index).resolve()
        if not source_index.is_file():
            raise FileNotFoundError(f"Missing validated source index: {source_index}")
        temporary = index_path.with_suffix(f".tmp-p{os.getpid()}")
        shutil.copyfile(source_index, temporary)
        os.replace(temporary, index_path)
        print(f"Copied validated dataset index: {source_index} -> {index_path}", flush=True)
    if world_size > 1:
        dist.barrier()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if args.max_samples > 0:
        payload = dict(payload)
        payload["samples"] = list(payload["samples"][: args.max_samples])
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


def load_official_vae(args: argparse.Namespace, device: torch.device):
    source = Path(args.official_repo).resolve() / "wan/modules/vae2_2.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing official Wan2.2 VAE source: {source}")
    spec = importlib.util.spec_from_file_location("wan22_official_vae", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official Wan2.2 VAE module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = Path(args.model_dir).resolve() / "Wan2.2_VAE.pth"
    return module.Wan2_2_VAE(
        vae_pth=str(checkpoint),
        dtype=torch.bfloat16,
        device=str(device),
    )


@torch.no_grad()
def encode_video(vae, video_tchw: torch.Tensor, device: torch.device) -> torch.Tensor:
    video_cthw = video_tchw.permute(1, 0, 2, 3).contiguous().to(torch.bfloat16)
    latent = vae.encode([video_cthw.to(device)])[0]
    if latent is None:
        raise RuntimeError("Official Wan2.2 VAE encode returned None")
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
    model_dir = Path(args.model_dir).resolve()
    official_vae_source = Path(args.official_repo).resolve() / "wan/modules/vae2_2.py"
    manifest_holder: list[dict | None] = [None]
    if rank == 0:
        clip_ids = [row["id"] for row in dataset.samples]
        contract = {
            "schema": CACHE_SCHEMA,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "model_dir": str(model_dir),
            "vae_checkpoint_sha256": sha256_file(model_dir / "Wan2.2_VAE.pth"),
            "official_vae_source": str(official_vae_source),
            "official_vae_source_sha256": sha256_file(official_vae_source),
            "source_dataset_index": str(Path(args.source_dataset_index).resolve()),
            "source_dataset_index_sha256": sha256_file(args.source_dataset_index),
            "ordered_clip_ids_hash": sha256_lines(clip_ids),
            "source_relative_paths_hash": sha256_lines(f"{clip_id}.pt" for clip_id in clip_ids),
            "builder_source_sha256": sha256_file(Path(__file__)),
            "dataset_source_sha256": sha256_file(THIS_DIR / "dataset.py"),
            "min_rgb_mean": args.min_rgb_mean,
            "num_samples": len(dataset),
            "scan_stats": dataset.scan_stats,
            "latent_shape": list(LATENT_SHAPE),
            "latent_dtype": "bfloat16",
            "prompt": dataset.prompt,
        }
        manifest = contract | {"cache_contract_hash": sha256_json(contract)}
        manifest_holder[0] = manifest
        complete = cache_dir / "CACHE_COMPLETE"
        if complete.exists():
            complete.unlink()
        manifest_path = cache_dir / "manifest.json"
        temporary = manifest_path.with_suffix(f".tmp-p{os.getpid()}")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, manifest_path)
        print(json.dumps(manifest, indent=2), flush=True)
    if world_size > 1:
        dist.broadcast_object_list(manifest_holder, src=0)
    manifest = manifest_holder[0]
    if not isinstance(manifest, dict):
        raise RuntimeError("Depth cache manifest broadcast failed")
    contract_hash = str(manifest["cache_contract_hash"])

    vae = load_official_vae(args, device)

    indices = range(rank, len(dataset), world_size)
    written = 0
    skipped = 0
    progress = tqdm(indices, desc=f"cache rank {rank}", disable=local_rank != 0)
    for index in progress:
        row = dataset.samples[index]
        destination = cache_path(cache_dir, row["id"])
        if existing_cache_is_valid(
            destination, clip_id=row["id"], prompt=dataset.prompt, contract_hash=contract_hash
        ):
            skipped += 1
            continue
        sample = dataset[index]
        source_latent = encode_video(vae, sample["source_video"], device)
        target_latent = encode_video(vae, sample["target_video"], device)
        payload = {
            "schema": CACHE_SCHEMA,
            "cache_contract_hash": contract_hash,
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
        actual = sorted(path.relative_to(cache_dir).as_posix() for path in cache_dir.glob("*/*.pt"))
        expected = sorted(f"{row['id']}.pt" for row in dataset.samples)
        if actual != expected:
            raise RuntimeError("Depth cache path set differs from the ordered dataset contract")
        manifest_path = cache_dir / "manifest.json"
        (cache_dir / "CACHE_COMPLETE").write_text(
            json.dumps(
                {
                    "schema": CACHE_SCHEMA,
                    "cache_contract_hash": contract_hash,
                    "manifest_sha256": sha256_file(manifest_path),
                    "num_samples": len(actual),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"CACHE COMPLETE: {len(actual)} samples", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
