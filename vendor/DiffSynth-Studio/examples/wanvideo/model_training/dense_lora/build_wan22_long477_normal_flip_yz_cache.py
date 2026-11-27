"""Encode 477-frame normals after [x,y,z] -> [x,-y,-z] conversion."""

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

CACHE_SCHEMA = "wan22_long477_normal_flip_yz_latent_v1"
INDEX_SCHEMA = "wan22_long477_source_index_v1"
NUM_FRAMES, HEIGHT, WIDTH = 477, 480, 832
LATENT_SHAPE = (48, 120, 30, 52)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--source_index", required=True)
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


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-p{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-p{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_official_vae(args: argparse.Namespace, device: torch.device):
    source = Path(args.official_repo).resolve() / "wan/modules/vae2_2.py"
    spec = importlib.util.spec_from_file_location("wan22_official_vae_flip_yz", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official Wan2.2 VAE module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = Path(args.model_dir).resolve() / "Wan2.2_VAE.pth"
    return module.Wan2_2_VAE(vae_pth=str(checkpoint), dtype=torch.bfloat16, device=str(device))


def load_normal_flip_yz(clip_dir: Path) -> torch.Tensor:
    video = torch.empty((NUM_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.bfloat16)
    for index in range(NUM_FRAMES):
        path = clip_dir / "normal" / f"{index + 1:06d}.exr"
        value = _read_normal(path)
        if value.shape != (HEIGHT, WIDTH, 3) or not np.isfinite(value).all():
            raise ValueError(f"Invalid normal frame {value.shape}: {path}")
        value[..., 1:] *= -1.0
        video[index].copy_(torch.from_numpy(value).permute(2, 0, 1))
    return video


@torch.no_grad()
def encode_video(vae, video_tchw: torch.Tensor, device: torch.device) -> torch.Tensor:
    latent = vae.encode([video_tchw.permute(1, 0, 2, 3).contiguous().to(device)])[0]
    if latent is None:
        raise RuntimeError("Official Wan2.2 VAE encode returned None")
    latent = latent.to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(latent.shape) != LATENT_SHAPE or not torch.isfinite(latent).all().item():
        raise ValueError(f"Invalid latent shape/dtype/values: {tuple(latent.shape)}, {latent.dtype}")
    return latent


def cache_path(cache_dir: Path, clip_id: str) -> Path:
    return cache_dir / "normal" / f"{clip_id}.pt"


def valid_cache(path: Path, clip_id: str, contract_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latent = payload["latent"]
        return bool(
            payload.get("schema") == CACHE_SCHEMA
            and payload.get("cache_contract_hash") == contract_hash
            and payload.get("clip_id") == clip_id
            and payload.get("modality") == "normal"
            and payload.get("normal_transform") == ["x", "-y", "-z"]
            and tuple(latent.shape) == LATENT_SHAPE
            and latent.dtype == torch.bfloat16
            and torch.isfinite(latent).all().item()
        )
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = init_distributed()
    source_index_path = Path(args.source_index).resolve()
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    if source_index.get("schema") != INDEX_SCHEMA:
        raise ValueError(f"Unsupported source index schema: {source_index.get('schema')}")
    if Path(source_index.get("dataset_root", "")).resolve() != Path(args.dataset_root).resolve():
        raise ValueError("Source index dataset root mismatch")
    rows = list(source_index["samples"])
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    holder: list[dict | None] = [None]
    if rank == 0:
        model_dir = Path(args.model_dir).resolve()
        official_source = Path(args.official_repo).resolve() / "wan/modules/vae2_2.py"
        contract = {
            "schema": CACHE_SCHEMA,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "source_index": str(source_index_path),
            "source_index_sha256": sha256_file(source_index_path),
            "num_clips": len(rows),
            "ordered_clip_ids_hash": sha256_lines(row["id"] for row in rows),
            "selected_frames": [1, NUM_FRAMES],
            "input_shape_tchw": [NUM_FRAMES, 3, HEIGHT, WIDTH],
            "latent_shape_cthw": list(LATENT_SHAPE),
            "latent_dtype": "bfloat16",
            "normal_input_coordinates": source_index.get("normal_input_coordinates", "opencv_camera_from_metadata"),
            "normal_transform_before_vae": ["x", "-y", "-z"],
            "vae_checkpoint_sha256": sha256_file(model_dir / "Wan2.2_VAE.pth"),
            "official_vae_source": str(official_source),
            "official_vae_source_sha256": sha256_file(official_source),
            "builder_sha256": sha256_file(__file__),
        }
        manifest = contract | {"cache_contract_hash": sha256_json(contract)}
        holder[0] = manifest
        complete = cache_dir / "CACHE_COMPLETE"
        if complete.exists():
            complete.unlink()
        atomic_json(cache_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if world_size > 1:
        dist.broadcast_object_list(holder, src=0)
    manifest = holder[0]
    if not isinstance(manifest, dict):
        raise RuntimeError("Manifest broadcast failed")
    contract_hash = str(manifest["cache_contract_hash"])

    vae = load_official_vae(args, device)
    written = skipped = 0
    for item in tqdm(range(rank, len(rows), world_size), desc=f"normal flip-yz rank {rank}", disable=local_rank != 0):
        row = rows[item]
        clip_id = row["id"]
        destination = cache_path(cache_dir, clip_id)
        if valid_cache(destination, clip_id, contract_hash):
            skipped += 1
            continue
        video = load_normal_flip_yz(Path(args.dataset_root).resolve() / clip_id)
        latent = encode_video(vae, video, device)
        atomic_torch_save(destination, {
            "schema": CACHE_SCHEMA,
            "cache_contract_hash": contract_hash,
            "clip_id": clip_id,
            "modality": "normal",
            "task_id": 1,
            "prompt": "output: surface normal",
            "rgb_mean": float(row["rgb_mean"]),
            "selected_frames": [1, NUM_FRAMES],
            "normal_transform": ["x", "-y", "-z"],
            "latent": latent,
        })
        written += 1
        del video, latent
        torch.cuda.empty_cache()
    print(f"rank={rank} written={written} skipped={skipped}", flush=True)

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        expected = sorted(row["id"] for row in rows)
        actual = sorted(path.relative_to(cache_dir / "normal").with_suffix("").as_posix() for path in (cache_dir / "normal").glob("*/*.pt"))
        if actual != expected:
            raise RuntimeError(f"Normal cache path set differs: expected {len(expected)}, got {len(actual)}")
        atomic_json(cache_dir / "CACHE_COMPLETE", {
            "schema": CACHE_SCHEMA,
            "cache_contract_hash": contract_hash,
            "manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "num_clips": len(actual),
            "count": len(actual),
        })
        print(f"NORMAL FLIP-YZ CACHE COMPLETE: {len(actual)}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
