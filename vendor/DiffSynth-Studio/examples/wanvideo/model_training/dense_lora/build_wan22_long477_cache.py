"""Cache 477-frame RGB/depth/normal/matte videos with the official Wan2.2 VAE.

The source clips contain 481 aligned frames. This cache deliberately uses only
frames 1..477 and encodes each complete video (no 81-frame window splitting).
Each modality is stored separately so task-routed training reads one target.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cache_integrity import sha256_file, sha256_json, sha256_lines  # noqa: E402
from dataset import _read_depth_exr, _ray_distance_to_camera_z  # noqa: E402
from dense4_dataset import _read_matte, _read_normal  # noqa: E402

CACHE_SCHEMA = "wan22_long477_dense_latent_v1"
INDEX_SCHEMA = "wan22_long477_source_index_v1"
NUM_FRAMES, SOURCE_FRAMES, HEIGHT, WIDTH = 477, 481, 480, 832
LATENT_SHAPE = (48, 120, 30, 52)
MODALITIES = ("source", "depth", "normal", "matte")
TASK_TO_ID = {"depth": 0, "normal": 1, "matte": 2}
TASK_PROMPTS = {"depth": "output: depth", "normal": "output: surface normal"}
ANIMAL_LABELS = (
    ("crane", ("red-crownedcrane", "red_crowned_crane", "crane")),
    ("horse", ("horse",)), ("bear", ("ursus", "bear")),
    ("lion", ("lion",)), ("crow", ("crow",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--official_repo", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--min_rgb_mean", type=float, default=40.0)
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


def subject_label(metadata: dict, metadata_path: Path) -> str:
    shot = metadata.get("shot")
    actor_type = shot.get("actor_type") if isinstance(shot, dict) else None
    if actor_type == "human":
        return "person"
    if actor_type != "animal":
        raise ValueError(f"{metadata_path}: unsupported actor_type {actor_type!r}")
    character = metadata.get("character")
    character_path = str(character.get("path", "")).lower() if isinstance(character, dict) else ""
    for label, keywords in ANIMAL_LABELS:
        if any(keyword in character_path for keyword in keywords):
            return label
    raise ValueError(f"{metadata_path}: cannot map animal subject from {character_path!r}")


def first477_rgb_mean(quality_path: Path) -> float:
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    frames = quality.get("frames")
    if not isinstance(frames, dict):
        raise ValueError(f"{quality_path}: missing frames mapping")
    values = []
    for frame_id in range(1, NUM_FRAMES + 1):
        row = frames.get(str(frame_id))
        rgb = row.get("rgb") if isinstance(row, dict) else None
        if not isinstance(rgb, dict) or "mean" not in rgb:
            raise ValueError(f"{quality_path}: frame {frame_id} has no rgb.mean")
        values.append(float(rgb["mean"]))
    if not np.isfinite(values).all():
        raise ValueError(f"{quality_path}: non-finite rgb.mean")
    return float(np.mean(values))


def expected_paths(clip_dir: Path, modality: str, suffix: str) -> list[Path]:
    return [clip_dir / modality / f"{frame_id:06d}{suffix}" for frame_id in range(1, NUM_FRAMES + 1)]


def scan_dataset(root: Path, min_rgb_mean: float, max_samples: int) -> dict:
    rows: list[dict] = []
    stats = Counter()
    for clip_dir in sorted(root.glob("*/*")):
        stats["scanned"] += 1
        if not clip_dir.is_dir() or not (clip_dir / "DONE").is_file():
            stats["missing_done"] += 1
            continue
        metadata_path, quality_path = clip_dir / "metadata.json", clip_dir / "quality.json"
        if not metadata_path.is_file() or not quality_path.is_file():
            stats["missing_metadata"] += 1
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        video, frames_data = metadata.get("video", {}), metadata.get("frames_data")
        if (
            video.get("frames") != SOURCE_FRAMES or video.get("height") != HEIGHT
            or video.get("width") != WIDTH or not isinstance(frames_data, list)
            or len(frames_data) != SOURCE_FRAMES
            or [row.get("frame") for row in frames_data[:NUM_FRAMES]] != list(range(1, NUM_FRAMES + 1))
        ):
            stats["bad_metadata_contract"] += 1
            continue
        storage = {"rgb": ".png", "depth": ".exr", "normal": ".exr", "matte": ".png"}
        if any(len(list((clip_dir / mod).glob(f"*{suffix}"))) != SOURCE_FRAMES for mod, suffix in storage.items()):
            stats["bad_frame_count"] += 1
            continue
        if any(
            not (clip_dir / mod / f"{frame_id:06d}{suffix}").is_file()
            for mod, suffix in storage.items() for frame_id in (1, NUM_FRAMES, SOURCE_FRAMES)
        ):
            stats["bad_frame_names"] += 1
            continue
        rgb_mean = first477_rgb_mean(quality_path)
        if rgb_mean < min_rgb_mean:
            stats["dark"] += 1
            continue
        rows.append({
            "id": f"{clip_dir.parent.name}/{clip_dir.name}",
            "rgb_mean": rgb_mean,
            "subject_label": subject_label(metadata, metadata_path),
        })
        stats["accepted"] += 1
        if max_samples > 0 and len(rows) >= max_samples:
            break
    if not rows:
        raise ValueError(f"No valid clips under {root}; stats={dict(stats)}")
    return {
        "schema": INDEX_SCHEMA, "dataset_root": str(root),
        "source_frames": SOURCE_FRAMES, "selected_frames": [1, NUM_FRAMES],
        "height": HEIGHT, "width": WIDTH, "min_rgb_mean": min_rgb_mean,
        "scan_stats": dict(stats), "samples": rows,
    }


def load_or_build_index(args: argparse.Namespace, cache_dir: Path, rank: int, world_size: int) -> dict:
    index_path = cache_dir / "source_index.json"
    if rank == 0 and not index_path.is_file():
        payload = scan_dataset(Path(args.dataset_root).resolve(), args.min_rgb_mean, args.max_samples)
        atomic_json(index_path, payload)
        print(f"Wrote fixed source index with {len(payload['samples'])} clips", flush=True)
    if world_size > 1:
        dist.barrier()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema") != INDEX_SCHEMA:
        raise ValueError(f"Unsupported source index: {index_path}")
    if Path(payload.get("dataset_root", "")).resolve() != Path(args.dataset_root).resolve():
        raise ValueError("Source index dataset_root mismatch")
    if float(payload.get("min_rgb_mean", -1.0)) != float(args.min_rgb_mean):
        raise ValueError("Source index brightness threshold mismatch")
    if args.max_samples > 0 and len(payload["samples"]) != args.max_samples:
        raise ValueError("Existing source index max_samples mismatch; use a separate cache directory")
    return payload


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
    return module.Wan2_2_VAE(vae_pth=str(checkpoint), dtype=torch.bfloat16, device=str(device))


def load_rgb(clip_dir: Path) -> torch.Tensor:
    video = torch.empty((NUM_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.bfloat16)
    for index, path in enumerate(expected_paths(clip_dir, "rgb", ".png")):
        with Image.open(path) as image:
            value = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if value.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"Bad RGB shape {value.shape}: {path}")
        video[index].copy_(torch.from_numpy(value).permute(2, 0, 1))
    video.div_(127.5).sub_(1.0)
    return video


def load_depth(clip_dir: Path, metadata: dict) -> tuple[torch.Tensor, float, float]:
    depth_type = metadata.get("coordinates", {}).get("depth_type")
    if depth_type != "blender_z_ray_distance_meters":
        raise ValueError(f"{clip_dir}: unsupported depth_type {depth_type!r}")
    frames_data = metadata["frames_data"][:NUM_FRAMES]
    inverse = np.empty((NUM_FRAMES, HEIGHT, WIDTH), dtype=np.float32)
    for index, (path, frame) in enumerate(zip(expected_paths(clip_dir, "depth", ".exr"), frames_data)):
        ray_distance = _read_depth_exr(path)
        if ray_distance.shape != (HEIGHT, WIDTH):
            raise ValueError(f"Bad depth shape {ray_distance.shape}: {path}")
        camera_z = _ray_distance_to_camera_z(ray_distance, frame["K"])
        if not np.isfinite(camera_z).all() or np.any(camera_z <= 0.0):
            raise ValueError(f"Invalid camera-Z pixels: {path}")
        np.reciprocal(camera_z, out=inverse[index])
    p2, p98 = (float(value) for value in np.percentile(inverse, (2.0, 98.0)))
    if not np.isfinite((p2, p98)).all() or p98 <= p2:
        raise ValueError(f"{clip_dir}: invalid inverse-depth percentiles {p2}, {p98}")
    np.subtract(inverse, p2, out=inverse)
    np.divide(inverse, p98 - p2, out=inverse)
    np.clip(inverse, 0.0, 1.0, out=inverse)
    scalar = torch.from_numpy(inverse).to(torch.bfloat16)
    video = scalar.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    video.mul_(2.0).sub_(1.0)
    return video, p2, p98


def load_normal(clip_dir: Path) -> torch.Tensor:
    video = torch.empty((NUM_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.bfloat16)
    for index, path in enumerate(expected_paths(clip_dir, "normal", ".exr")):
        value = _read_normal(path)
        if value.shape != (HEIGHT, WIDTH, 3) or not np.isfinite(value).all():
            raise ValueError(f"Bad normal frame {value.shape}: {path}")
        video[index].copy_(torch.from_numpy(value).permute(2, 0, 1))
    return video


def load_matte(clip_dir: Path) -> torch.Tensor:
    video = torch.empty((NUM_FRAMES, 3, HEIGHT, WIDTH), dtype=torch.bfloat16)
    for index, path in enumerate(expected_paths(clip_dir, "matte", ".png")):
        value = _read_matte(path)
        if value.shape != (HEIGHT, WIDTH) or not np.isfinite(value).all():
            raise ValueError(f"Bad matte frame {value.shape}: {path}")
        video[index].copy_(torch.from_numpy(value).to(torch.bfloat16).unsqueeze(0).expand(3, -1, -1))
    video.mul_(2.0).sub_(1.0)
    return video


@torch.no_grad()
def encode_video(vae, video_tchw: torch.Tensor, device: torch.device) -> torch.Tensor:
    video_cthw = video_tchw.permute(1, 0, 2, 3).contiguous()
    latent = vae.encode([video_cthw.to(device)])[0]
    if latent is None:
        raise RuntimeError("Official Wan2.2 VAE encode returned None")
    latent = latent.to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(latent.shape) != LATENT_SHAPE or not torch.isfinite(latent).all().item():
        raise ValueError(f"Invalid latent: shape={tuple(latent.shape)}, dtype={latent.dtype}")
    return latent


def cache_path(cache_dir: Path, modality: str, clip_id: str) -> Path:
    scene, clip = clip_id.split("/", 1)
    return cache_dir / modality / scene / f"{clip}.pt"


def valid_cache(path: Path, clip_id: str, modality: str, contract_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latent = payload["latent"]
        return bool(
            payload.get("schema") == CACHE_SCHEMA
            and payload.get("cache_contract_hash") == contract_hash
            and payload.get("clip_id") == clip_id and payload.get("modality") == modality
            and payload.get("task_id") == TASK_TO_ID.get(modality)
            and tuple(latent.shape) == LATENT_SHAPE and latent.dtype == torch.bfloat16
            and torch.isfinite(latent).all().item()
        )
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def prompt_for(modality: str, label: str) -> str | None:
    if modality == "source":
        return None
    if modality == "matte":
        return f"output: {label} alpha matte"
    return TASK_PROMPTS[modality]


def build_manifest(args: argparse.Namespace, cache_dir: Path, index: dict) -> dict:
    model_dir = Path(args.model_dir).resolve()
    official_source = Path(args.official_repo).resolve() / "wan/modules/vae2_2.py"
    clip_ids = [row["id"] for row in index["samples"]]
    contract = {
        "schema": CACHE_SCHEMA, "source_index_schema": INDEX_SCHEMA,
        "dataset_root": str(Path(args.dataset_root).resolve()), "model_dir": str(model_dir),
        "num_clips": len(clip_ids), "ordered_clip_ids_hash": sha256_lines(clip_ids),
        "source_index_sha256": sha256_file(cache_dir / "source_index.json"),
        "vae_checkpoint_sha256": sha256_file(model_dir / "Wan2.2_VAE.pth"),
        "official_vae_source": str(official_source),
        "official_vae_source_sha256": sha256_file(official_source),
        "builder_sha256": sha256_file(__file__), "modalities": list(MODALITIES),
        "selected_frames": [1, NUM_FRAMES], "input_shape_tchw": [NUM_FRAMES, 3, HEIGHT, WIDTH],
        "latent_shape_cthw": list(LATENT_SHAPE), "latent_dtype": "bfloat16",
        "depth_preprocess": "ray-distance to camera-Z; inverse; clip-global p2/p98 over frames 1..477; [0,1] to [-1,1]",
        "min_rgb_mean": args.min_rgb_mean,
        "subject_counts": dict(sorted(Counter(row["subject_label"] for row in index["samples"]).items())),
    }
    return contract | {"cache_contract_hash": sha256_json(contract)}


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = init_distributed()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_index = load_or_build_index(args, cache_dir, rank, world_size)
    rows = source_index["samples"]
    holder: list[dict | None] = [None]
    if rank == 0:
        manifest = build_manifest(args, cache_dir, source_index)
        holder[0] = manifest
        complete = cache_dir / "CACHE_COMPLETE"
        if complete.exists(): complete.unlink()
        atomic_json(cache_dir / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if world_size > 1:
        dist.broadcast_object_list(holder, src=0)
    manifest = holder[0]
    if not isinstance(manifest, dict): raise RuntimeError("Manifest broadcast failed")
    contract_hash = str(manifest["cache_contract_hash"])

    vae = load_official_vae(args, device)
    written = skipped = 0
    for index_number in tqdm(range(rank, len(rows), world_size), desc=f"long477 rank {rank}", disable=local_rank != 0):
        row, clip_id = rows[index_number], rows[index_number]["id"]
        destinations = {modality: cache_path(cache_dir, modality, clip_id) for modality in MODALITIES}
        missing = [m for m, path in destinations.items() if not valid_cache(path, clip_id, m, contract_hash)]
        if not missing:
            skipped += len(MODALITIES)
            continue
        clip_dir = Path(args.dataset_root).resolve() / clip_id
        metadata = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
        depth_stats: dict[str, float] = {}
        for modality in missing:
            if modality == "source": video = load_rgb(clip_dir)
            elif modality == "depth":
                video, p2, p98 = load_depth(clip_dir, metadata)
                depth_stats = {"depth_p2": p2, "depth_p98": p98}
            elif modality == "normal": video = load_normal(clip_dir)
            elif modality == "matte": video = load_matte(clip_dir)
            else: raise AssertionError(modality)
            latent = encode_video(vae, video, device)
            payload = {
                "schema": CACHE_SCHEMA, "cache_contract_hash": contract_hash,
                "clip_id": clip_id, "modality": modality, "task_id": TASK_TO_ID.get(modality),
                "prompt": prompt_for(modality, row["subject_label"]),
                "subject_label": row["subject_label"] if modality == "matte" else None,
                "rgb_mean": float(row["rgb_mean"]), "selected_frames": [1, NUM_FRAMES], "latent": latent,
            }
            if modality == "depth": payload.update(depth_stats)
            atomic_torch_save(destinations[modality], payload)
            written += 1
            del video, latent, payload
            torch.cuda.empty_cache()
        atomic_json(cache_dir / "metadata" / f"{clip_id}.json", {
            "schema": CACHE_SCHEMA, "cache_contract_hash": contract_hash, "clip_id": clip_id,
            "source_clip_dir": str(clip_dir.resolve()), "rgb_mean": float(row["rgb_mean"]),
            "subject_label": row["subject_label"], "selected_frames": [1, NUM_FRAMES], **depth_stats,
        })
    print(f"rank={rank} written={written} skipped={skipped}", flush=True)

    if world_size > 1: dist.barrier()
    if rank == 0:
        expected = sorted(row["id"] for row in rows)
        counts = {}
        for modality in MODALITIES:
            actual = sorted(p.relative_to(cache_dir / modality).with_suffix("").as_posix() for p in (cache_dir / modality).glob("*/*.pt"))
            if actual != expected:
                raise RuntimeError(f"{modality} cache paths differ: expected {len(expected)}, got {len(actual)}")
            counts[modality] = len(actual)
        metadata_ids = sorted(p.relative_to(cache_dir / "metadata").with_suffix("").as_posix() for p in (cache_dir / "metadata").glob("*/*.json"))
        if metadata_ids != expected:
            raise RuntimeError(f"metadata paths differ: expected {len(expected)}, got {len(metadata_ids)}")
        atomic_json(cache_dir / "CACHE_COMPLETE", {
            "schema": CACHE_SCHEMA, "cache_contract_hash": contract_hash,
            "manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "num_clips": len(expected), "counts": counts,
        })
        print(f"LONG477 CACHE COMPLETE: {counts}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
