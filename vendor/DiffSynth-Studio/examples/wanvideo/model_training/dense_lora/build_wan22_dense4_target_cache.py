"""Build Wan2.2 normal/matte/raymap VAE target caches.

Readers and task semantics match the reviewed dense4 integration. Encoding uses
the official Wan2.2 ``Wan2_2_VAE`` implementation and TI2V-5B checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dense4_dataset import Dense4TargetDataset  # noqa: E402
from cache_integrity import (  # noqa: E402
    READER_SCHEMA,
    expected_depth_paths_hash,
    expected_task_paths_hash,
    prompt_manifest_hash,
    sha256_file,
    sha256_json,
    sha256_lines,
)


CACHE_SCHEMA = "wan22_dense4_target_latent_v2"
DEPTH_SCHEMA = "wan22_depth_latent_v2"
LATENT_SHAPE = (48, 21, 30, 52)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--prompt_dir", required=True)
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
        # Cache workers do not exchange tensors. Gloo is sufficient for the
        # manifest broadcast/barriers and avoids unnecessary NCCL dependency.
        dist.init_process_group("gloo")
    return rank, local_rank, world_size, device


def target_path(cache_dir: Path, task: str, clip_id: str) -> Path:
    scene, clip = clip_id.split("/", 1)
    return cache_dir / task / scene / f"{clip}.pt"


def valid_cache(path: Path, *, clip_id: str, task: str, contract_hash: str) -> bool:
    """Fully validate an existing payload before allowing a resume skip."""
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latent = payload["target_latent"]
        return bool(
            payload.get("schema") == CACHE_SCHEMA
            and payload.get("reader_schema") == READER_SCHEMA
            and payload.get("cache_contract_hash") == contract_hash
            and payload.get("clip_id") == clip_id
            and payload.get("task") == task
            and int(payload.get("task_id", -1)) == {"normal": 1, "matte": 2, "raymap": 3}[task]
            and tuple(latent.shape) == LATENT_SHAPE
            and latent.dtype == torch.bfloat16
            and torch.isfinite(latent).all().item()
        )
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


@torch.no_grad()
def encode_targets(
    vae,
    targets: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    videos = [target.permute(1, 0, 2, 3).contiguous().to(torch.bfloat16) for target in targets]
    result = []
    for video in videos:
        latent = vae.encode([video.to(device)])[0]
        if latent is None:
            raise RuntimeError("Official Wan2.2 VAE encode returned None")
        latent = latent.to(device="cpu", dtype=torch.bfloat16).contiguous()
        if tuple(latent.shape) != LATENT_SHAPE or not torch.isfinite(latent).all():
            raise ValueError(f"Invalid target latent {tuple(latent.shape)}")
        result.append(latent)
    return result


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


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = init_distributed()
    depth_cache = Path(args.depth_cache_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dense4TargetDataset(
        args.dataset_root,
        depth_cache / "dataset_index.json",
        args.prompt_dir,
        max_samples=args.max_samples,
    )
    clip_ids = [row["id"] for row in dataset.samples]
    model_dir = Path(args.model_dir).resolve()
    official_vae_source = Path(args.official_repo).resolve() / "wan/modules/vae2_2.py"
    depth_index_path = depth_cache / "dataset_index.json"
    depth_manifest_path = depth_cache / "manifest.json"
    if not depth_manifest_path.is_file():
        raise FileNotFoundError(f"Missing source cache manifest: {depth_manifest_path}")
    depth_complete_path = depth_cache / "CACHE_COMPLETE"
    if not depth_complete_path.is_file():
        raise FileNotFoundError(f"Incomplete source cache: {depth_cache}")
    manifest_holder: list[dict | None] = [None]
    if rank == 0:
        depth_manifest = json.loads(depth_manifest_path.read_text(encoding="utf-8"))
        depth_complete = json.loads(depth_complete_path.read_text(encoding="utf-8"))
        if depth_manifest.get("schema") != DEPTH_SCHEMA or depth_complete.get("schema") != DEPTH_SCHEMA:
            raise ValueError("Dense4 v2 targets require a Wan2.2 depth v2 cache")
        if depth_complete.get("cache_contract_hash") != depth_manifest.get("cache_contract_hash"):
            raise ValueError("Depth cache contract hash mismatch")
        if depth_complete.get("manifest_sha256") != sha256_file(depth_manifest_path):
            raise ValueError("Depth cache manifest hash mismatch")
        actual_depth_paths = sorted(
            path.relative_to(depth_cache).as_posix() for path in depth_cache.glob("*/*.pt")
        )
        expected_depth_paths = sorted(f"{clip_id}.pt" for clip_id in clip_ids)
        if actual_depth_paths != expected_depth_paths:
            raise ValueError("Depth cache path set differs from the Dense4 clip contract")
        subject_counts = Counter(row["subject_label"] for row in dataset.samples)
        task_path_hashes = {
            task: expected_task_paths_hash(task, clip_ids)
            for task in dataset.target_tasks
        }
        contract = {
            "schema": CACHE_SCHEMA,
            "reader_schema": READER_SCHEMA,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "depth_cache_dir": str(depth_cache),
            "model_dir": str(model_dir),
            "num_clips": len(dataset),
            "ordered_clip_ids_hash": sha256_lines(clip_ids),
            "depth_dataset_index_hash": sha256_file(depth_index_path),
            "source_cache_manifest_hash": sha256_file(depth_manifest_path),
            "source_cache_contract_hash": depth_manifest["cache_contract_hash"],
            "source_relative_paths_hash": expected_depth_paths_hash(clip_ids),
            "vae_checkpoint_sha256": sha256_file(model_dir / "Wan2.2_VAE.pth"),
            "official_vae_source": str(official_vae_source),
            "official_vae_source_sha256": sha256_file(official_vae_source),
            "prompt_manifest_hash": prompt_manifest_hash(args.prompt_dir),
            "task_relative_paths_hash": task_path_hashes,
            "tasks": list(dataset.target_tasks),
            "task_samples": len(dataset) * len(dataset.target_tasks),
            "min_rgb_mean": 40.0,
            "subject_counts": dict(sorted(subject_counts.items())),
            "latent_shape": list(LATENT_SHAPE),
            "latent_dtype": "bfloat16",
        }
        manifest = contract | {"cache_contract_hash": sha256_json(contract)}
        manifest_holder[0] = manifest
        complete = cache_dir / "CACHE_COMPLETE"
        if complete.exists():
            complete.unlink()
        (cache_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2), flush=True)
    if world_size > 1:
        dist.broadcast_object_list(manifest_holder, src=0)
    manifest = manifest_holder[0]
    if not isinstance(manifest, dict):
        raise RuntimeError("Dense4 manifest broadcast failed")
    contract_hash = str(manifest["cache_contract_hash"])

    vae = load_official_vae(args, device)

    indices = range(rank, len(dataset), world_size)
    written = skipped = 0
    for index in tqdm(indices, desc=f"dense4 cache rank {rank}", disable=local_rank != 0):
        row = dataset.samples[index]
        destinations = {
            task: target_path(cache_dir, task, row["id"])
            for task in dataset.target_tasks
        }
        missing_tasks = [
            task for task, path in destinations.items()
            if not valid_cache(path, clip_id=row["id"], task=task, contract_hash=contract_hash)
        ]
        if not missing_tasks:
            skipped += len(destinations)
            continue
        sample = dataset[index]
        latents = encode_targets(
            vae,
            [sample["targets"][task] for task in missing_tasks],
            device,
        )
        for task, latent in zip(missing_tasks, latents):
            destination = destinations[task]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".tmp-r{rank}-p{os.getpid()}")
            torch.save(
                {
                    "schema": CACHE_SCHEMA,
                    "reader_schema": READER_SCHEMA,
                    "cache_contract_hash": contract_hash,
                    "clip_id": sample["clip_id"],
                    "task": task,
                    "task_id": {"normal": 1, "matte": 2, "raymap": 3}[task],
                    "prompt": sample["prompts"][task],
                    "subject_label": sample["subject_label"] if task == "matte" else None,
                    "rgb_mean": float(sample["rgb_mean"]),
                    "target_latent": latent,
                },
                temporary,
            )
            os.replace(temporary, destination)
            written += 1
        del sample, latents

    print(f"rank={rank} written={written} skipped={skipped}", flush=True)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        counts = {}
        for task in dataset.target_tasks:
            actual = sorted(path.relative_to(cache_dir).as_posix() for path in (cache_dir / task).glob("*/*.pt"))
            expected = sorted(f"{task}/{clip_id}.pt" for clip_id in clip_ids)
            if actual != expected:
                raise RuntimeError(f"Dense4 {task} cache path set differs from expected clip IDs")
            counts[task] = len(actual)
        if any(count != len(dataset) for count in counts.values()):
            raise RuntimeError(f"Dense4 target cache incomplete: expected={len(dataset)}, counts={counts}")
        (cache_dir / "CACHE_COMPLETE").write_text(
            json.dumps(
                {
                    "schema": CACHE_SCHEMA,
                    "cache_contract_hash": contract_hash,
                    "manifest_sha256": sha256_file(cache_dir / "manifest.json"),
                    "num_clips": len(dataset),
                    "counts": counts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"DENSE4 TARGET CACHE COMPLETE: {counts}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
