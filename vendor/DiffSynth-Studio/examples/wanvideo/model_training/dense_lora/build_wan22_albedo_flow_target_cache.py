"""Build Wan2.2 albedo and one-frame-delayed optical-flow target caches.

The raw flow contract is forward pixel displacement ``t -> t+1`` stored as
EXR ``R=u, G=v, B=valid``.  The causal target sequence is shifted by one RGB
frame: target frame 1 is invalid and target frame t stores raw flow t-1 -> t.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

from wan22_cache_integrity import sha256_file, sha256_json, sha256_lines  # noqa: E402


os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

CACHE_SCHEMA = "wan22_albedo_delayed_flow_target_latent_v1"
READER_SCHEMA = "wan22_albedo_delayed_flow_reader_v1"
SOURCE_SCHEMAS = {"wan22_depth_latent_v1", "wan22_depth_latent_v2"}
TASK_TO_ID = {"albedo": 4, "optical_flow": 5}
TASK_PROMPTS = {
    "albedo": "output: albedo",
    "optical_flow": "output: optical flow",
}
NUM_FRAMES = 81
HEIGHT = 480
WIDTH = 832
LATENT_SHAPE = (48, 21, 30, 52)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--source_cache_dir", required=True)
    parser.add_argument("--official_repo", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def init_distributed() -> tuple[int, int, int, torch.device | None]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = None
    if not args_validate_only():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("gloo")
    return rank, local_rank, world_size, device


def args_validate_only() -> bool:
    return "--validate_only" in sys.argv


def target_path(root: Path, task: str, clip_id: str) -> Path:
    scene, clip = clip_id.split("/", 1)
    return root / task / scene / f"{clip}.pt"


def _read_exr(path: Path) -> np.ndarray:
    import cv2

    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise ValueError(f"OpenCV failed to read EXR: {path}")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"EXR contains NaN/Inf: {path}")
    return value


def load_albedo(clip_dir: Path) -> tuple[torch.Tensor, int, int]:
    frames = []
    clipped = total = 0
    for frame_id in range(1, NUM_FRAMES + 1):
        path = clip_dir / "albedo" / f"{frame_id:06d}.exr"
        packed = _read_exr(path)
        if packed.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"Unexpected albedo shape {packed.shape}: {path}")
        rgb = packed[..., ::-1].copy()  # OpenCV BGR -> RGB.
        clipped += int(np.count_nonzero((rgb < 0.0) | (rgb > 1.0)))
        total += int(rgb.size)
        frames.append(np.clip(rgb, 0.0, 1.0))
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    video = video.mul_(2.0).sub_(1.0)
    if tuple(video.shape) != (NUM_FRAMES, 3, HEIGHT, WIDTH):
        raise ValueError(f"Unexpected albedo video shape: {tuple(video.shape)}")
    return video, clipped, total


def load_delayed_flow(clip_dir: Path) -> tuple[torch.Tensor, dict[str, float]]:
    # Output channels are normalized u/W, v/H, and signed validity.  Frame 1
    # is invalid; frame t consumes raw forward flow t-1 -> t.
    output = np.zeros((NUM_FRAMES, HEIGHT, WIDTH, 3), dtype=np.float32)
    output[0, ..., 2] = -1.0
    clipped_u = clipped_v = valid_count = 0
    max_magnitude = 0.0
    for output_index in range(1, NUM_FRAMES):
        raw_frame_id = output_index
        path = clip_dir / "optical_flow" / f"{raw_frame_id:06d}.exr"
        packed = _read_exr(path)
        if packed.shape != (HEIGHT, WIDTH, 4):
            raise ValueError(f"Unexpected optical-flow shape {packed.shape}: {path}")
        # OpenCV reads the declared RGBA EXR as BGRA.
        u = packed[..., 2]
        v = packed[..., 1]
        valid = packed[..., 0] > 0.5
        if np.any(np.abs(packed[..., 0] - valid.astype(np.float32)) > 1.0e-6):
            raise ValueError(f"Non-binary optical-flow validity: {path}")
        if valid.any():
            max_magnitude = max(
                max_magnitude,
                float(np.sqrt(np.square(u[valid]) + np.square(v[valid])).max()),
            )
        invalid_nonzero = (~valid) & ((np.abs(u) > 1.0e-6) | (np.abs(v) > 1.0e-6))
        if invalid_nonzero.any():
            raise ValueError(f"Invalid optical-flow pixels contain nonzero vectors: {path}")
        clipped_u += int(np.count_nonzero(valid & (np.abs(u) > WIDTH)))
        clipped_v += int(np.count_nonzero(valid & (np.abs(v) > HEIGHT)))
        valid_count += int(valid.sum())
        output[output_index, ..., 0] = np.clip(u / float(WIDTH), -1.0, 1.0)
        output[output_index, ..., 1] = np.clip(v / float(HEIGHT), -1.0, 1.0)
        output[output_index, ..., 2] = valid.astype(np.float32) * 2.0 - 1.0
    video = torch.from_numpy(output).permute(0, 3, 1, 2).contiguous()
    return video, {
        "valid_count": float(valid_count),
        "clipped_u_count": float(clipped_u),
        "clipped_v_count": float(clipped_v),
        "max_magnitude_px": float(max_magnitude),
    }


def load_official_vae(args: argparse.Namespace, device: torch.device):
    source = Path(args.official_repo).resolve() / "wan/modules/vae2_2.py"
    spec = importlib.util.spec_from_file_location("wan22_official_vae", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official Wan2.2 VAE module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Wan2_2_VAE(
        vae_pth=str(Path(args.model_dir).resolve() / "Wan2.2_VAE.pth"),
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
        raise ValueError(f"Invalid target latent: {tuple(latent.shape)}")
    return latent


def valid_cache(path: Path, *, clip_id: str, task: str, contract_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latent = payload.get("target_latent")
        return bool(
            payload.get("schema") == CACHE_SCHEMA
            and payload.get("reader_schema") == READER_SCHEMA
            and payload.get("cache_contract_hash") == contract_hash
            and payload.get("clip_id") == clip_id
            and payload.get("task") == task
            and int(payload.get("task_id", -1)) == TASK_TO_ID[task]
            and payload.get("prompt") == TASK_PROMPTS[task]
            and isinstance(latent, torch.Tensor)
            and tuple(latent.shape) == LATENT_SHAPE
            and latent.dtype == torch.bfloat16
            and torch.isfinite(latent).all().item()
        )
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def validate_clip_contract(dataset_root: Path, clip_ids: list[str], rank: int, world_size: int) -> None:
    failures = []
    expected_names = {f"{frame_id:06d}.exr" for frame_id in range(1, NUM_FRAMES + 1)}
    for clip_id in clip_ids[rank::world_size]:
        clip_dir = dataset_root / clip_id
        marker_path = clip_dir / ".albedo_flow_complete.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if int(marker.get("frames", -1)) != NUM_FRAMES:
                raise ValueError("completion marker frame count differs")
            for task in ("albedo", "optical_flow"):
                directory = clip_dir / task
                actual_names = {
                    entry.name
                    for entry in os.scandir(directory)
                    if entry.is_file() and not entry.name.startswith(".")
                }
                if actual_names != expected_names:
                    missing = sorted(expected_names - actual_names)[:3]
                    extra = sorted(actual_names - expected_names)[:3]
                    raise ValueError(f"{task} frame names differ; missing={missing}, extra={extra}")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"{clip_id}: {error}")
            if len(failures) >= 8:
                break
    if failures:
        raise RuntimeError("Raw albedo/flow contract validation failed: " + "; ".join(failures))


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = init_distributed()
    dataset_root = Path(args.dataset_root).resolve()
    source_root = Path(args.source_cache_dir).resolve()
    cache_root = Path(args.cache_dir).resolve()
    source_index_path = source_root / "dataset_index.json"
    source_manifest_path = source_root / "manifest.json"
    source_complete_path = source_root / "CACHE_COMPLETE"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_complete = json.loads(source_complete_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema") not in SOURCE_SCHEMAS:
        raise ValueError(f"Unsupported source cache schema: {source_manifest.get('schema')}")
    if int(source_complete.get("num_samples", -1)) != int(source_manifest.get("num_samples", -2)):
        raise ValueError("Source cache complete/manifest sample counts differ")
    if Path(source_manifest.get("dataset_root", "")).resolve() != dataset_root:
        raise ValueError("Source cache dataset root differs")
    clip_ids = [str(row["id"]) for row in source_index["samples"]]
    if args.max_samples > 0:
        clip_ids = clip_ids[: args.max_samples]
    if not clip_ids or len(clip_ids) != len(set(clip_ids)):
        raise ValueError("Source index contains no clips or duplicate IDs")

    validate_clip_contract(dataset_root, clip_ids, rank, world_size)
    if world_size > 1:
        dist.barrier()
    if args.validate_only:
        if rank == 0:
            print(json.dumps({"validated_clips": len(clip_ids), "tasks": list(TASK_TO_ID)}), flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return

    if device is None:
        raise RuntimeError("CUDA device was not initialized")
    cache_root.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_dir).resolve()
    official_vae_source = Path(args.official_repo).resolve() / "wan/modules/vae2_2.py"
    contract = {
        "schema": CACHE_SCHEMA,
        "reader_schema": READER_SCHEMA,
        "dataset_root": str(dataset_root),
        "source_cache_dir": str(source_root),
        "source_cache_schema": source_manifest["schema"],
        "source_dataset_index_sha256": sha256_file(source_index_path),
        "source_cache_manifest_sha256": sha256_file(source_manifest_path),
        "ordered_clip_ids_hash": sha256_lines(clip_ids),
        "num_clips": len(clip_ids),
        "tasks": list(TASK_TO_ID),
        "task_ids": TASK_TO_ID,
        "task_prompts": TASK_PROMPTS,
        "model_dir": str(model_root),
        "vae_checkpoint_sha256": sha256_file(model_root / "Wan2.2_VAE.pth"),
        "official_vae_source": str(official_vae_source),
        "official_vae_source_sha256": sha256_file(official_vae_source),
        "builder_source_sha256": sha256_file(Path(__file__)),
        "raw_flow_contract": "R=u,G=v,B=valid; forward pixels t->t+1; +x right,+y down; frame81 invalid",
        "target_flow_alignment": "target frame1 invalid; target frame t stores raw flow t-1->t for t=2..81",
        "flow_tensor_contract": "channels=(clip(u/832,-1,1),clip(v/480,-1,1),2*valid-1)",
        "albedo_tensor_contract": "linear Cycles Diffuse Color RGB clipped to [0,1], then 2*x-1",
        "latent_shape": list(LATENT_SHAPE),
        "latent_dtype": "bfloat16",
    }
    manifest = contract | {"cache_contract_hash": sha256_json(contract)}
    contract_hash = str(manifest["cache_contract_hash"])
    if rank == 0:
        complete_path = cache_root / "CACHE_COMPLETE"
        if complete_path.exists():
            complete_path.unlink()
        manifest_path = cache_root / "manifest.json"
        temporary = manifest_path.with_suffix(f".tmp-p{os.getpid()}")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, manifest_path)
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if world_size > 1:
        dist.barrier()

    vae = load_official_vae(args, device)
    written = skipped = 0
    for position in tqdm(range(rank, len(clip_ids), world_size), desc=f"albedo-flow rank {rank}", disable=local_rank != 0):
        clip_id = clip_ids[position]
        destinations = {task: target_path(cache_root, task, clip_id) for task in TASK_TO_ID}
        missing = [task for task, path in destinations.items() if not valid_cache(path, clip_id=clip_id, task=task, contract_hash=contract_hash)]
        if not missing:
            skipped += len(TASK_TO_ID)
            continue
        clip_dir = dataset_root / clip_id
        albedo_video = flow_video = None
        metadata: dict[str, object] = {}
        if "albedo" in missing:
            albedo_video, clipped, total = load_albedo(clip_dir)
            metadata["albedo_clipped_count"] = clipped
            metadata["albedo_value_count"] = total
        if "optical_flow" in missing:
            flow_video, flow_stats = load_delayed_flow(clip_dir)
            metadata["flow_stats"] = flow_stats
        for task, video in (("albedo", albedo_video), ("optical_flow", flow_video)):
            if task not in missing:
                continue
            if video is None:
                raise RuntimeError(f"Missing decoded video for {task}/{clip_id}")
            latent = encode_video(vae, video, device)
            destination = destinations[task]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".tmp-r{rank}-p{os.getpid()}")
            torch.save(
                {
                    "schema": CACHE_SCHEMA,
                    "reader_schema": READER_SCHEMA,
                    "cache_contract_hash": contract_hash,
                    "clip_id": clip_id,
                    "task": task,
                    "task_id": TASK_TO_ID[task],
                    "prompt": TASK_PROMPTS[task],
                    "target_latent": latent,
                    "preprocess_metadata": metadata,
                },
                temporary,
            )
            os.replace(temporary, destination)
            written += 1
    print(f"rank={rank} written={written} skipped={skipped}", flush=True)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        counts = {}
        for task in TASK_TO_ID:
            actual = sorted(path.relative_to(cache_root).as_posix() for path in (cache_root / task).glob("*/*.pt"))
            expected = sorted(f"{task}/{clip_id}.pt" for clip_id in clip_ids)
            if actual != expected:
                raise RuntimeError(f"{task} cache path set differs from expected clip IDs")
            counts[task] = len(actual)
        complete = {
            "schema": CACHE_SCHEMA,
            "cache_contract_hash": contract_hash,
            "manifest_sha256": sha256_file(cache_root / "manifest.json"),
            "num_clips": len(clip_ids),
            "counts": counts,
        }
        (cache_root / "CACHE_COMPLETE").write_text(json.dumps(complete, indent=2), encoding="utf-8")
        print(f"ALBEDO/FLOW CACHE COMPLETE: {counts}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
