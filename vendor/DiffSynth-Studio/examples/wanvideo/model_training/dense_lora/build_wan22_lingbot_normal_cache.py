#!/usr/bin/env python3
"""Build a Wan2.2 VAE latent cache for LingBot surface normals."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cache_integrity import sha256_file, sha256_json, sha256_lines  # noqa: E402
from dense4_dataset import _read_normal  # noqa: E402


CACHE_SCHEMA = "wan22_lingbot_normal_latent_v1"
DEPTH_SCHEMA = "wan22_depth_latent_v2"
NORMAL_SOURCE_SCHEMA = "lingbot-wan-normal81-v2"
LATENT_SHAPE = (48, 21, 30, 52)
PROMPT = "output: surface normal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--official_repo", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("gloo")
    return rank, local_rank, world_size, device


def cache_path(root: Path, clip_id: str) -> Path:
    scene, clip = clip_id.split("/", 1)
    return root / scene / f"{clip}.pt"


def valid_latent(value: object) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == LATENT_SHAPE
        and value.dtype == torch.bfloat16
        and torch.isfinite(value).all().item()
    )


def valid_cache(path: Path, clip_id: str, contract_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return bool(
            payload.get("schema") == CACHE_SCHEMA
            and payload.get("cache_contract_hash") == contract_hash
            and payload.get("clip_id") == clip_id
            and payload.get("task") == "normal"
            and payload.get("task_id") == 1
            and payload.get("prompt") == PROMPT
            and valid_latent(payload.get("target_latent"))
        )
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def load_normal_video(dataset_root: Path, clip_id: str) -> torch.Tensor:
    clip = dataset_root / clip_id
    marker = json.loads((clip / "NORMAL_DONE").read_text(encoding="utf-8"))
    if marker.get("schema") != NORMAL_SOURCE_SCHEMA:
        raise ValueError(f"Unsupported/incomplete source normal: {clip}")
    frames = []
    for frame in range(1, 82):
        value = _read_normal(clip / "normal" / f"{frame:06d}.png")
        if value.shape != (480, 832, 3) or not torch.from_numpy(value).isfinite().all():
            raise ValueError(f"Invalid normal frame: {clip}/{frame:06d}")
        frames.append(value)
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    if video.shape != (81, 3, 480, 832) or not torch.isfinite(video).all():
        raise ValueError(f"Invalid normal video: {clip_id}")
    return video


def load_vae(args: argparse.Namespace, device: torch.device):
    source = Path(args.official_repo).resolve() / "wan/modules/vae2_2.py"
    spec = importlib.util.spec_from_file_location("wan22_official_vae", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official Wan2.2 VAE: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Wan2_2_VAE(
        vae_pth=str(Path(args.model_dir).resolve() / "Wan2.2_VAE.pth"),
        dtype=torch.bfloat16,
        device=str(device),
    )


@torch.no_grad()
def encode(vae, video: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = video.permute(1, 0, 2, 3).contiguous().to(torch.bfloat16)
    latent = vae.encode([value.to(device)])[0]
    if latent is None:
        raise RuntimeError("Wan2.2 VAE returned None")
    latent = latent.to(device="cpu", dtype=torch.bfloat16).contiguous()
    if not valid_latent(latent):
        raise ValueError(f"Invalid normal latent: {tuple(latent.shape)}/{latent.dtype}")
    return latent


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = init_distributed()
    dataset_root = Path(args.dataset_root).resolve()
    depth_cache = Path(args.depth_cache_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    depth_manifest_path = depth_cache / "manifest.json"
    depth_complete_path = depth_cache / "CACHE_COMPLETE"
    depth_index_path = depth_cache / "dataset_index.json"
    for path in (depth_manifest_path, depth_complete_path, depth_index_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    depth_manifest = json.loads(depth_manifest_path.read_text(encoding="utf-8"))
    depth_complete = json.loads(depth_complete_path.read_text(encoding="utf-8"))
    if depth_manifest.get("schema") != DEPTH_SCHEMA or depth_complete.get("schema") != DEPTH_SCHEMA:
        raise ValueError("Normal cache requires Wan2.2 depth v2 cache")
    if depth_complete.get("cache_contract_hash") != depth_manifest.get("cache_contract_hash"):
        raise ValueError("Depth cache contract mismatch")
    if depth_complete.get("manifest_sha256") != sha256_file(depth_manifest_path):
        raise ValueError("Depth manifest hash mismatch")
    index = json.loads(depth_index_path.read_text(encoding="utf-8"))
    samples = list(index["samples"])
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    clip_ids = [row["id"] for row in samples]

    manifest_holder: list[dict | None] = [None]
    if rank == 0:
        source_manifest = dataset_root.parent / "normal_manifest.json"
        source_validation = dataset_root.parent / "normal_validation_report.json"
        contract = {
            "schema": CACHE_SCHEMA,
            "dataset_root": str(dataset_root),
            "depth_cache_dir": str(depth_cache),
            "source_cache_contract_hash": depth_manifest["cache_contract_hash"],
            "depth_manifest_sha256": sha256_file(depth_manifest_path),
            "depth_dataset_index_sha256": sha256_file(depth_index_path),
            "normal_source_manifest_sha256": sha256_file(source_manifest),
            "normal_source_validation_sha256": sha256_file(source_validation),
            "model_dir": str(Path(args.model_dir).resolve()),
            "vae_checkpoint_sha256": sha256_file(Path(args.model_dir).resolve() / "Wan2.2_VAE.pth"),
            "official_vae_source_sha256": sha256_file(Path(args.official_repo).resolve() / "wan/modules/vae2_2.py"),
            "builder_source_sha256": sha256_file(Path(__file__)),
            "reader_source_sha256": sha256_file(THIS_DIR / "dense4_dataset.py"),
            "num_samples": len(samples),
            "ordered_clip_ids_hash": sha256_lines(clip_ids),
            "relative_paths_hash": sha256_lines(f"{clip_id}.pt" for clip_id in clip_ids),
            "task": "normal",
            "task_id": 1,
            "prompt": PROMPT,
            "latent_shape": list(LATENT_SHAPE),
            "latent_dtype": "bfloat16",
        }
        manifest = contract | {"cache_contract_hash": sha256_json(contract)}
        manifest_holder[0] = manifest
        complete = cache_dir / "CACHE_COMPLETE"
        if complete.exists():
            complete.unlink()
        (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if world_size > 1:
        dist.broadcast_object_list(manifest_holder, src=0)
    manifest = manifest_holder[0]
    if not isinstance(manifest, dict):
        raise RuntimeError("Normal cache manifest broadcast failed")
    contract_hash = str(manifest["cache_contract_hash"])
    vae = load_vae(args, device)

    written = skipped = 0
    for index_value in tqdm(range(rank, len(samples), world_size), desc=f"normal rank {rank}", disable=local_rank != 0):
        clip_id = samples[index_value]["id"]
        destination = cache_path(cache_dir, clip_id)
        if valid_cache(destination, clip_id, contract_hash):
            skipped += 1
            continue
        video = load_normal_video(dataset_root, clip_id)
        latent = encode(vae, video, device)
        payload = {
            "schema": CACHE_SCHEMA,
            "cache_contract_hash": contract_hash,
            "clip_id": clip_id,
            "task": "normal",
            "task_id": 1,
            "prompt": PROMPT,
            "target_latent": latent,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".tmp-r{rank}-p{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        written += 1
        del video, latent, payload
    print(f"rank={rank} written={written} skipped={skipped}", flush=True)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        actual = sorted(path.relative_to(cache_dir).as_posix() for path in cache_dir.glob("*/*.pt"))
        expected = sorted(f"{clip_id}.pt" for clip_id in clip_ids)
        if actual != expected:
            raise RuntimeError("Normal cache path set differs from dataset index")
        (cache_dir / "CACHE_COMPLETE").write_text(
            json.dumps({
                "schema": CACHE_SCHEMA,
                "cache_contract_hash": contract_hash,
                "manifest_sha256": sha256_file(cache_dir / "manifest.json"),
                "num_samples": len(actual),
            }, indent=2) + "\n"
        )
        print(f"CACHE COMPLETE: {len(actual)} normal samples", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
