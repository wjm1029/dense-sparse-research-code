#!/usr/bin/env python3
"""Build compact latent-grid validity weights from LingBot hole masks.

The original depth/normal videos were nearest-filled before VAE encoding.  This
cache preserves the pre-fill sensor validity information for the latent MSE.
Each weight is the valid-pixel fraction in the corresponding causal 4-frame x
16 x 16 VAE downsampling cell; a fully invalid cell therefore contributes zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
from cache_integrity import sha256_file, sha256_json, sha256_lines  # noqa: E402


SCHEMA = "wan22_lingbot_latent_validity_v1"
DEPTH_SCHEMA = "wan22_depth_latent_v2"
MASK_SHAPE = (1, 21, 30, 52)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def cache_path(root: Path, clip_id: str) -> Path:
    scene, clip = clip_id.split("/", 1)
    return root / scene / f"{clip}.pt"


def source_mask_path(dataset_root: Path, clip_id: str, frame: int) -> Path:
    return dataset_root / clip_id / "valid_mask" / f"{frame:06d}.png"


def build_one(job: tuple[str, str]) -> dict:
    clip_id, dataset_root_value = job
    dataset_root = Path(dataset_root_value)
    masks = []
    source_hash = hashlib.sha256()
    for frame in range(1, 82):
        path = source_mask_path(dataset_root, clip_id, frame)
        with Image.open(path) as image:
            value = np.asarray(image)
        if value.shape != (480, 832) or value.dtype != np.uint8 or not np.isin(value, (0, 255)).all():
            raise ValueError(f"Invalid binary validity mask: {path} {value.shape}/{value.dtype}")
        valid = value > 0
        source_hash.update(np.packbits(valid, axis=None).tobytes())
        masks.append(valid)

    pixel = np.stack(masks).reshape(81, 30, 16, 52, 16)
    spatial = pixel.mean(axis=(2, 4), dtype=np.float32)
    temporal = np.concatenate(
        (spatial[:1], spatial[1:].reshape(20, 4, 30, 52).mean(axis=1, dtype=np.float32)),
        axis=0,
    )
    latent_mask = torch.from_numpy(temporal[None]).to(torch.float16).contiguous()
    if tuple(latent_mask.shape) != MASK_SHAPE or not torch.isfinite(latent_mask).all():
        raise ValueError(f"Invalid latent mask for {clip_id}: {tuple(latent_mask.shape)}")
    return {
        "clip_id": clip_id,
        "latent_validity": latent_mask,
        "source_mask_sha256": source_hash.hexdigest(),
        "pixel_valid_fraction": float(pixel.mean()),
        "latent_valid_fraction": float(latent_mask.float().mean()),
    }


def valid_existing(path: Path, result: dict, contract_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mask = payload.get("latent_validity")
        return bool(
            payload.get("schema") == SCHEMA
            and payload.get("cache_contract_hash") == contract_hash
            and payload.get("clip_id") == result["clip_id"]
            and payload.get("source_mask_sha256") == result["source_mask_sha256"]
            and isinstance(mask, torch.Tensor)
            and tuple(mask.shape) == MASK_SHAPE
            and mask.dtype == torch.float16
            and torch.isfinite(mask).all().item()
            and torch.equal(mask, result["latent_validity"])
        )
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    depth_root = Path(args.depth_cache_dir).resolve()
    cache_root = Path(args.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    depth_manifest_path = depth_root / "manifest.json"
    depth_complete_path = depth_root / "CACHE_COMPLETE"
    depth_index_path = depth_root / "dataset_index.json"
    for path in (depth_manifest_path, depth_complete_path, depth_index_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    depth_manifest = json.loads(depth_manifest_path.read_text(encoding="utf-8"))
    depth_complete = json.loads(depth_complete_path.read_text(encoding="utf-8"))
    if depth_manifest.get("schema") != DEPTH_SCHEMA or depth_complete.get("schema") != DEPTH_SCHEMA:
        raise ValueError("Mask cache requires a complete Wan2.2 depth-v2 cache")
    if depth_complete.get("manifest_sha256") != sha256_file(depth_manifest_path):
        raise ValueError("Depth manifest hash mismatch")
    index = json.loads(depth_index_path.read_text(encoding="utf-8"))
    samples = list(index["samples"])
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    clip_ids = [row["id"] for row in samples]
    if len(clip_ids) != len(set(clip_ids)) or not clip_ids:
        raise ValueError("Depth index clip IDs must be non-empty and unique")
    dataset_root = Path(depth_manifest["dataset_root"]).resolve()

    complete_path = cache_root / "CACHE_COMPLETE"
    if complete_path.exists():
        complete_path.unlink()
    results: dict[str, dict] = {}
    jobs = [(clip_id, str(dataset_root)) for clip_id in clip_ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results[result["clip_id"]] = result
            if count % 100 == 0 or count == len(futures):
                print(f"read_masks={count}/{len(futures)}", flush=True)

    source_lines = [f"{clip_id}\t{results[clip_id]['source_mask_sha256']}" for clip_id in clip_ids]
    contract = {
        "schema": SCHEMA,
        "depth_cache_dir": str(depth_root),
        "depth_cache_contract_hash": depth_manifest["cache_contract_hash"],
        "depth_manifest_sha256": sha256_file(depth_manifest_path),
        "depth_dataset_index_sha256": sha256_file(depth_index_path),
        "dataset_root": str(dataset_root),
        "num_samples": len(clip_ids),
        "ordered_clip_ids_hash": sha256_lines(clip_ids),
        "relative_paths_hash": sha256_lines(f"{clip_id}.pt" for clip_id in clip_ids),
        "source_masks_sha256": sha256_lines(source_lines),
        "builder_source_sha256": sha256_file(Path(__file__)),
        "source_shape": [81, 480, 832],
        "mask_shape": list(MASK_SHAPE),
        "mask_dtype": "float16",
        "mask_semantics": "valid-pixel fraction per causal temporal4-spatial16 VAE cell",
        "temporal_mapping": "latent[0]=frame[0]; latent[i]=mean(frame[4*i-3:4*i+1]), i=1..20",
        "spatial_mapping": "non-overlapping mean over 16x16 pixels",
    }
    manifest = contract | {
        "cache_contract_hash": sha256_json(contract),
        "pixel_valid_fraction_mean": float(
            np.mean([results[clip_id]["pixel_valid_fraction"] for clip_id in clip_ids])
        ),
        "latent_valid_fraction_mean": float(
            np.mean([results[clip_id]["latent_valid_fraction"] for clip_id in clip_ids])
        ),
    }
    manifest_path = cache_root / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(f".tmp-p{os.getpid()}")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary_manifest, manifest_path)

    contract_hash = manifest["cache_contract_hash"]
    written = skipped = 0
    for count, clip_id in enumerate(clip_ids, 1):
        result = results[clip_id]
        destination = cache_path(cache_root, clip_id)
        if valid_existing(destination, result, contract_hash):
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "cache_contract_hash": contract_hash,
            "clip_id": clip_id,
            "source_mask_sha256": result["source_mask_sha256"],
            "pixel_valid_fraction": result["pixel_valid_fraction"],
            "latent_valid_fraction": result["latent_valid_fraction"],
            "latent_validity": result["latent_validity"],
        }
        temporary = destination.with_suffix(f".tmp-p{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        written += 1
        if count % 250 == 0:
            print(f"write_masks={count}/{len(clip_ids)}", flush=True)

    expected = sorted(f"{clip_id}.pt" for clip_id in clip_ids)
    actual = sorted(path.relative_to(cache_root).as_posix() for path in cache_root.glob("*/*.pt"))
    if actual != expected:
        raise RuntimeError("Mask cache path set differs from depth index")
    complete = {
        "schema": SCHEMA,
        "cache_contract_hash": contract_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "num_samples": len(actual),
    }
    temporary_complete = complete_path.with_suffix(f".tmp-p{os.getpid()}")
    temporary_complete.write_text(json.dumps(complete, indent=2) + "\n")
    os.replace(temporary_complete, complete_path)
    print(json.dumps({"written": written, "skipped": skipped, **complete}, indent=2), flush=True)


if __name__ == "__main__":
    main()
