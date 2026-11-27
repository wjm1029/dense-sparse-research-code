"""Six-task Wan2.2-TI2V-5B LoRA training with a task-aware rank HyperNet."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.pipelines.wan_video import (
    ModelConfig,
    WanVideoPipeline,
    WanVideoUnit_PromptEmbedder,
)


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from wan22_dense6_task_router import (  # noqa: E402
    ID_TO_TASK,
    TASK_INSTRUCTIONS,
    TASK_TO_ID,
    TaskHyperRouter,
    discover_lora_sites,
)
from wan22_cache_integrity import (  # noqa: E402
    READER_SCHEMA,
    expected_depth_paths_hash,
    expected_task_paths_hash,
    prompt_manifest_hash,
    require_equal,
    sha256_file,
    sha256_json,
    sha256_lines,
)
from wan22_dense6_checkpoint import (  # noqa: E402
    build_provenance_metadata,
    load_dense6_checkpoint,
    save_dense6_checkpoint,
)
from wan22_dense6_task_sampler import (  # noqa: E402
    SixTaskStratifiedDistributedSampler,
    ProportionalTaskPoolDistributedSampler,
)
from wan22_masked_loss import masked_latent_mse  # noqa: E402


DEPTH_SCHEMA = "wan22_depth_latent_v2"
TARGET_SCHEMA = "wan22_dense4_target_latent_v2"
LEGACY_DEPTH_SCHEMA = "wan22_depth_latent_v1"
LEGACY_TARGET_SCHEMA = "wan22_dense4_target_latent_v1"
EXTRA_TARGET_SCHEMA = "wan22_albedo_delayed_flow_target_latent_v1"
EXTRA_READER_SCHEMA = "wan22_albedo_delayed_flow_reader_v1"
LATENT_SHAPE = (48, 21, 30, 52)
MASK_SHAPE = (1, 21, 30, 52)
LINGBOT_NORMAL_SCHEMA = "wan22_lingbot_normal_latent_v1"
LINGBOT_MASK_SCHEMA = "wan22_lingbot_latent_validity_v1"


def broadcast_object_list(values: list, *, src: int) -> None:
    """Use CUDA staging for NCCL (required by PyTorch 2.12), CPU for Gloo."""
    if dist.get_backend() == "nccl":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        dist.broadcast_object_list(values, src=src, device=device)
    else:
        dist.broadcast_object_list(values, src=src)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--target_cache_dir", required=True)
    parser.add_argument("--extra_target_cache_dir", required=True)
    parser.add_argument("--expected_flow_scale", type=float, default=0.0,
                        help="Require a fixed pixel-scale flow contract when nonzero.")
    parser.add_argument("--lingbot_depth_cache_dir", default=None)
    parser.add_argument("--lingbot_normal_cache_dir", default=None)
    parser.add_argument("--lingbot_mask_cache_dir", default=None)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--prompt_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--code_patch_path", required=True)
    parser.add_argument(
        "--init_code_patch_path", default=None,
        help="Immutable source patch associated with --init_checkpoint (defaults to current patch).",
    )
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--router_learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--router_weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--router_identity_weight", type=float, default=1.0e-3)
    parser.add_argument("--router_mode", choices=("learned", "identity"), default="learned")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--router_hidden_dim", type=int, default=512)
    parser.add_argument("--lora_target_modules", default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument(
        "--init_checkpoint", default=None,
        help="Initialize LoRA/router weights only; optimizer, scheduler, RNG and step start fresh.",
    )
    parser.add_argument("--allow_existing_output", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--allow_legacy_v1_cache",
        action="store_true",
        help="Trust the existing Wan2.2 v1 cache manifests, then fully audit every payload.",
    )
    parser.add_argument("--distributed_backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--router_freeze_step", type=int, default=-1)
    parser.add_argument(
        "--single_gpu_smoke",
        action="store_true",
        help="Explicitly permit a one-GPU contract/smoke run; formal training remains four-GPU only.",
    )
    return parser.parse_args()


def depth_path(root: Path, clip_id: str) -> Path:
    scene, clip = clip_id.split("/", 1)
    return root / scene / f"{clip}.pt"


def target_path(root: Path, task: str, clip_id: str) -> Path:
    scene, clip = clip_id.split("/", 1)
    return root / task / scene / f"{clip}.pt"


def validate_cache_contract(args: argparse.Namespace, rank: int, world_size: int) -> None:
    """Recompute provenance hashes on rank 0 and fail all ranks consistently."""
    error_holder: list[str | None] = [None]
    if rank == 0:
        try:
            depth_root = Path(args.depth_cache_dir).resolve()
            target_root = Path(args.target_cache_dir).resolve()
            extra_root = Path(args.extra_target_cache_dir).resolve()
            manifest_path = target_root / "manifest.json"
            complete_path = target_root / "CACHE_COMPLETE"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            target_schema = LEGACY_TARGET_SCHEMA if args.allow_legacy_v1_cache else TARGET_SCHEMA
            depth_schema = LEGACY_DEPTH_SCHEMA if args.allow_legacy_v1_cache else DEPTH_SCHEMA
            require_equal(manifest.get("schema"), target_schema, "target schema")
            require_equal(
                manifest.get("cache_contract_hash"),
                sha256_json({key: value for key, value in manifest.items() if key != "cache_contract_hash"}),
                "target contract content hash",
            )
            require_equal(manifest.get("reader_schema"), READER_SCHEMA, "reader schema")
            require_equal(manifest.get("dataset_root"), str(Path(args.dataset_root).resolve()), "dataset root")
            require_equal(manifest.get("depth_cache_dir"), str(depth_root), "depth cache root")
            require_equal(manifest.get("model_dir"), str(Path(args.model_dir).resolve()), "model root")
            require_equal(complete.get("schema"), target_schema, "complete schema")
            require_equal(complete.get("cache_contract_hash"), manifest.get("cache_contract_hash"), "contract hash")
            require_equal(complete.get("manifest_sha256"), sha256_file(manifest_path), "manifest hash")

            depth_index_path = depth_root / "dataset_index.json"
            depth_manifest_path = depth_root / "manifest.json"
            depth_complete_path = depth_root / "CACHE_COMPLETE"
            depth_index = json.loads(depth_index_path.read_text(encoding="utf-8"))
            depth_manifest = json.loads(depth_manifest_path.read_text(encoding="utf-8"))
            depth_complete = json.loads(depth_complete_path.read_text(encoding="utf-8"))
            require_equal(depth_manifest.get("schema"), depth_schema, "depth schema")
            require_equal(depth_complete.get("schema"), depth_schema, "depth complete schema")
            if args.allow_legacy_v1_cache:
                require_equal(depth_complete.get("num_samples"), int(manifest["num_clips"]), "depth count")
            else:
                require_equal(
                    depth_manifest.get("cache_contract_hash"),
                    sha256_json({key: value for key, value in depth_manifest.items() if key != "cache_contract_hash"}),
                    "depth contract content hash",
                )
                require_equal(
                    depth_complete.get("cache_contract_hash"), depth_manifest.get("cache_contract_hash"),
                    "depth contract hash",
                )
                require_equal(
                    depth_complete.get("manifest_sha256"), sha256_file(depth_manifest_path),
                    "depth manifest hash",
                )
                require_equal(
                    manifest.get("source_cache_contract_hash"), depth_manifest.get("cache_contract_hash"),
                    "target-to-depth contract",
                )
            num_clips = int(manifest["num_clips"])
            clip_ids = [row["id"] for row in depth_index["samples"][:num_clips]]
            require_equal(len(clip_ids), num_clips, "ordered clip count")
            require_equal(manifest.get("ordered_clip_ids_hash"), sha256_lines(clip_ids), "ordered clip IDs")
            require_equal(manifest.get("depth_dataset_index_hash"), sha256_file(depth_index_path), "depth index")
            require_equal(manifest.get("source_cache_manifest_hash"), sha256_file(depth_manifest_path), "source manifest")
            require_equal(manifest.get("source_relative_paths_hash"), expected_depth_paths_hash(clip_ids), "source paths")
            actual_source_paths = sorted(
                path.relative_to(depth_root).as_posix() for path in depth_root.glob("*/*.pt")
            )
            expected_source_paths = sorted(f"{clip_id}.pt" for clip_id in clip_ids)
            missing_source_paths = sorted(set(expected_source_paths) - set(actual_source_paths))
            require_equal(missing_source_paths, [], "source expected path coverage")
            require_equal(
                manifest.get("vae_checkpoint_sha256"),
                sha256_file(Path(args.model_dir).resolve() / "Wan2.2_VAE.pth"),
                "VAE checkpoint",
            )
            require_equal(manifest.get("prompt_manifest_hash"), prompt_manifest_hash(args.prompt_dir), "prompts")
            for task in ("normal", "matte", "raymap"):
                require_equal(
                    manifest["task_relative_paths_hash"].get(task),
                    expected_task_paths_hash(task, clip_ids),
                    f"{task} paths",
                )
                actual_task_paths = sorted(
                    path.relative_to(target_root).as_posix()
                    for path in (target_root / task).glob("*/*.pt")
                )
                expected_task_paths = sorted(f"{task}/{clip_id}.pt" for clip_id in clip_ids)
                require_equal(actual_task_paths, expected_task_paths, f"{task} exact path set")

            extra_manifest_path = extra_root / "manifest.json"
            extra_complete_path = extra_root / "CACHE_COMPLETE"
            extra_manifest = json.loads(extra_manifest_path.read_text(encoding="utf-8"))
            extra_complete = json.loads(extra_complete_path.read_text(encoding="utf-8"))
            require_equal(extra_manifest.get("schema"), EXTRA_TARGET_SCHEMA, "extra target schema")
            require_equal(extra_manifest.get("reader_schema"), EXTRA_READER_SCHEMA, "extra reader schema")
            require_equal(
                extra_manifest.get("cache_contract_hash"),
                sha256_json({key: value for key, value in extra_manifest.items() if key != "cache_contract_hash"}),
                "extra target contract content hash",
            )
            require_equal(extra_complete.get("schema"), EXTRA_TARGET_SCHEMA, "extra complete schema")
            require_equal(
                extra_complete.get("cache_contract_hash"),
                extra_manifest.get("cache_contract_hash"),
                "extra contract hash",
            )
            require_equal(
                extra_complete.get("manifest_sha256"),
                sha256_file(extra_manifest_path),
                "extra manifest hash",
            )
            require_equal(extra_manifest.get("dataset_root"), str(Path(args.dataset_root).resolve()), "extra dataset root")
            require_equal(extra_manifest.get("source_cache_dir"), str(depth_root), "extra source cache root")
            require_equal(extra_manifest.get("source_dataset_index_sha256"), sha256_file(depth_index_path), "extra source index")
            require_equal(extra_manifest.get("source_cache_manifest_sha256"), sha256_file(depth_manifest_path), "extra source manifest")
            require_equal(extra_manifest.get("ordered_clip_ids_hash"), sha256_lines(clip_ids), "extra ordered clip IDs")
            require_equal(int(extra_manifest.get("num_clips", -1)), num_clips, "extra clip count")
            require_equal(extra_manifest.get("task_ids"), {"albedo": 4, "optical_flow": 5}, "extra task IDs")
            if args.expected_flow_scale > 0:
                scale = args.expected_flow_scale
                require_equal(extra_manifest.get("flow_scale_px"), [scale, scale], "flow pixel scale")
                require_equal(extra_manifest.get("flow_tensor_contract"),
                              f"channels=(clip(u/{scale:g},-1,1),clip(v/{scale:g},-1,1),2*valid-1)",
                              "flow normalization semantics")
            require_equal(
                extra_manifest.get("vae_checkpoint_sha256"),
                sha256_file(Path(args.model_dir).resolve() / "Wan2.2_VAE.pth"),
                "extra VAE checkpoint",
            )
            for task in ("albedo", "optical_flow"):
                actual_task_paths = sorted(
                    path.relative_to(extra_root).as_posix()
                    for path in (extra_root / task).glob("*/*.pt")
                )
                expected_task_paths = sorted(f"{task}/{clip_id}.pt" for clip_id in clip_ids)
                require_equal(actual_task_paths, expected_task_paths, f"{task} exact path set")

            auxiliary_values = (
                args.lingbot_depth_cache_dir,
                args.lingbot_normal_cache_dir,
                args.lingbot_mask_cache_dir,
            )
            if any(auxiliary_values) != all(auxiliary_values):
                raise ValueError("All three LingBot cache arguments must be supplied together")
            if all(auxiliary_values):
                aux_roots = [Path(value).resolve() for value in auxiliary_values]
                aux_manifests = []
                for root in aux_roots:
                    aux_manifest_path = root / "manifest.json"
                    aux_complete_path = root / "CACHE_COMPLETE"
                    aux_manifest = json.loads(aux_manifest_path.read_text(encoding="utf-8"))
                    aux_complete = json.loads(aux_complete_path.read_text(encoding="utf-8"))
                    require_equal(aux_complete.get("schema"), aux_manifest.get("schema"), f"{root} schema")
                    require_equal(
                        aux_complete.get("cache_contract_hash"), aux_manifest.get("cache_contract_hash"),
                        f"{root} contract",
                    )
                    require_equal(aux_complete.get("manifest_sha256"), sha256_file(aux_manifest_path), f"{root} manifest")
                    aux_manifests.append(aux_manifest)
                lingbot_depth, lingbot_normal, lingbot_mask = aux_manifests
                require_equal(lingbot_depth.get("schema"), DEPTH_SCHEMA, "LingBot depth schema")
                require_equal(lingbot_normal.get("schema"), LINGBOT_NORMAL_SCHEMA, "LingBot normal schema")
                require_equal(lingbot_mask.get("schema"), LINGBOT_MASK_SCHEMA, "LingBot mask schema")
                depth_contract = lingbot_depth.get("cache_contract_hash")
                require_equal(lingbot_normal.get("source_cache_contract_hash"), depth_contract, "normal-to-depth contract")
                require_equal(lingbot_mask.get("depth_cache_contract_hash"), depth_contract, "mask-to-depth contract")
                ordered_hash = lingbot_depth.get("ordered_clip_ids_hash")
                require_equal(lingbot_normal.get("ordered_clip_ids_hash"), ordered_hash, "normal clip ordering")
                require_equal(lingbot_mask.get("ordered_clip_ids_hash"), ordered_hash, "mask clip ordering")
                require_equal(lingbot_normal.get("task_id"), TASK_TO_ID["normal"], "normal task ID")
                lingbot_index = json.loads((aux_roots[0] / "dataset_index.json").read_text(encoding="utf-8"))
                lingbot_ids = [row["id"] for row in lingbot_index["samples"]]
                require_equal(sha256_lines(lingbot_ids), ordered_hash, "LingBot depth index ordering")
                for root in aux_roots:
                    actual = sorted(path.relative_to(root).as_posix() for path in root.glob("*/*.pt"))
                    expected = sorted(f"{clip_id}.pt" for clip_id in lingbot_ids)
                    require_equal(actual, expected, f"{root} exact payload paths")
        except Exception as error:  # Broadcast a readable failure instead of hanging peers.
            error_holder[0] = f"{type(error).__name__}: {error}"
    if world_size > 1:
        broadcast_object_list(error_holder, src=0)
    if error_holder[0] is not None:
        raise RuntimeError(f"Dense6 cache contract validation failed: {error_holder[0]}")


def _finite_bf16_latent(value: object) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == LATENT_SHAPE
        and value.dtype == torch.bfloat16
        and torch.isfinite(value).all().item()
    )


class CachedDense6Dataset(Dataset):
    """Original Dense6 plus masked LingBot depth/normal samples."""

    def __init__(
        self, depth_cache_dir: str | Path, target_cache_dir: str | Path,
        extra_target_cache_dir: str | Path,
        *, allow_legacy_v1_cache: bool = False,
        lingbot_depth_cache_dir: str | Path | None = None,
        lingbot_normal_cache_dir: str | Path | None = None,
        lingbot_mask_cache_dir: str | Path | None = None,
    ) -> None:
        self.depth_root = Path(depth_cache_dir).resolve()
        self.target_root = Path(target_cache_dir).resolve()
        self.extra_target_root = Path(extra_target_cache_dir).resolve()
        for root in (self.depth_root, self.target_root, self.extra_target_root):
            if not (root / "CACHE_COMPLETE").is_file():
                raise FileNotFoundError(f"Cache is incomplete: {root}")
        target_manifest = json.loads((self.target_root / "manifest.json").read_text(encoding="utf-8"))
        self.target_schema = LEGACY_TARGET_SCHEMA if allow_legacy_v1_cache else TARGET_SCHEMA
        self.depth_schema = LEGACY_DEPTH_SCHEMA if allow_legacy_v1_cache else DEPTH_SCHEMA
        if target_manifest.get("schema") != self.target_schema:
            raise ValueError("Unsupported dense6 target cache")
        self.num_clips = int(target_manifest["num_clips"])
        self.cache_contract_hash = str(target_manifest["cache_contract_hash"])
        extra_manifest = json.loads(
            (self.extra_target_root / "manifest.json").read_text(encoding="utf-8")
        )
        if extra_manifest.get("schema") != EXTRA_TARGET_SCHEMA:
            raise ValueError("Unsupported albedo/flow target cache")
        self.extra_cache_contract_hash = str(extra_manifest["cache_contract_hash"])
        depth_manifest = json.loads((self.depth_root / "manifest.json").read_text(encoding="utf-8"))
        if depth_manifest.get("schema") != self.depth_schema:
            raise ValueError("Unsupported depth cache schema")
        self.depth_contract_hash = depth_manifest.get("cache_contract_hash")
        if (not allow_legacy_v1_cache
                and target_manifest.get("source_cache_contract_hash") != self.depth_contract_hash):
            raise ValueError("Target cache does not bind the active depth cache contract")
        depth_index = json.loads((self.depth_root / "dataset_index.json").read_text(encoding="utf-8"))
        self.clip_ids = [row["id"] for row in depth_index["samples"][: self.num_clips]]
        if len(self.clip_ids) != self.num_clips:
            raise ValueError("Depth index has fewer clips than the target cache")
        if sha256_lines(self.clip_ids) != target_manifest.get("ordered_clip_ids_hash"):
            raise ValueError("Target cache ordered clip IDs do not match the depth index")
        if (
            int(extra_manifest.get("num_clips", -1)) != self.num_clips
            or extra_manifest.get("ordered_clip_ids_hash") != sha256_lines(self.clip_ids)
        ):
            raise ValueError("Albedo/flow cache clip IDs do not match the depth index")
        subjects = sorted(target_manifest["subject_counts"])
        self.prompt_texts = [
            "output: depth",
            "output: surface normal",
            "output: camera raymap",
            "output: albedo",
            "output: optical flow",
            *[f"output: {subject} alpha matte" for subject in subjects],
        ]
        self.prompt_to_id = {prompt: index for index, prompt in enumerate(self.prompt_texts)}
        auxiliary = (lingbot_depth_cache_dir, lingbot_normal_cache_dir, lingbot_mask_cache_dir)
        if any(auxiliary) != all(auxiliary):
            raise ValueError("All three LingBot cache directories must be supplied together")
        self.lingbot_roots: tuple[Path, Path, Path] | None = None
        self.lingbot_clip_ids: list[str] = []
        self.lingbot_contracts: tuple[str, str, str] | None = None
        if all(auxiliary):
            roots = tuple(Path(value).resolve() for value in auxiliary)
            manifests = [json.loads((root / "manifest.json").read_text(encoding="utf-8")) for root in roots]
            index = json.loads((roots[0] / "dataset_index.json").read_text(encoding="utf-8"))
            self.lingbot_clip_ids = [row["id"] for row in index["samples"]]
            self.lingbot_roots = roots
            self.lingbot_contracts = tuple(str(item["cache_contract_hash"]) for item in manifests)
        base_count = self.num_clips * len(TASK_TO_ID)
        lingbot_count = len(self.lingbot_clip_ids)
        self.task_pools = {
            task_id: [
                clip_index * len(TASK_TO_ID) + task_id
                for clip_index in range(self.num_clips)
            ]
            for task_id in range(len(TASK_TO_ID))
        }
        self.task_pools[TASK_TO_ID["depth"]].extend(base_count + index for index in range(lingbot_count))
        self.task_pools[TASK_TO_ID["normal"]].extend(
            base_count + lingbot_count + index for index in range(lingbot_count)
        )

    def target_contract(self, task: str) -> tuple[Path, str, str, str]:
        if task in ("albedo", "optical_flow"):
            return (
                self.extra_target_root,
                EXTRA_TARGET_SCHEMA,
                EXTRA_READER_SCHEMA,
                self.extra_cache_contract_hash,
            )
        return self.target_root, self.target_schema, READER_SCHEMA, self.cache_contract_hash

    def validate_all_payloads(self, rank: int, world_size: int) -> None:
        """Distributed full deserialization audit before optimizer creation."""
        local_count = 0
        local_error: str | None = None
        try:
            for clip_index in range(rank, self.num_clips, world_size):
                clip_id = self.clip_ids[clip_index]
                source_payload = torch.load(
                    depth_path(self.depth_root, clip_id), map_location="cpu", weights_only=False
                )
                if (
                    source_payload.get("schema") != self.depth_schema
                    or (self.depth_contract_hash is not None
                        and source_payload.get("cache_contract_hash") != self.depth_contract_hash)
                    or source_payload.get("clip_id") != clip_id
                    or not _finite_bf16_latent(source_payload.get("source_latent"))
                    or not _finite_bf16_latent(source_payload.get("target_latent"))
                ):
                    raise ValueError(f"Invalid source cache payload: {clip_id}")
                local_count += 1
                for task_id in range(1, len(TASK_TO_ID)):
                    task = ID_TO_TASK[task_id]
                    task_root, task_schema, reader_schema, contract_hash = self.target_contract(task)
                    payload = torch.load(
                        target_path(task_root, task, clip_id), map_location="cpu", weights_only=False
                    )
                    if (
                        payload.get("schema") != task_schema
                        or payload.get("reader_schema") != reader_schema
                        or payload.get("cache_contract_hash") != contract_hash
                        or payload.get("clip_id") != clip_id
                        or payload.get("task") != task
                        or int(payload.get("task_id", -1)) != task_id
                        or not _finite_bf16_latent(payload.get("target_latent"))
                    ):
                        raise ValueError(f"Invalid {task} target cache payload: {clip_id}")
                    local_count += 1
            if self.lingbot_roots is not None and self.lingbot_contracts is not None:
                depth_root, normal_root, mask_root = self.lingbot_roots
                depth_contract, normal_contract, mask_contract = self.lingbot_contracts
                for clip_index in range(rank, len(self.lingbot_clip_ids), world_size):
                    clip_id = self.lingbot_clip_ids[clip_index]
                    source_payload = torch.load(depth_path(depth_root, clip_id), map_location="cpu", weights_only=False)
                    normal_payload = torch.load(depth_path(normal_root, clip_id), map_location="cpu", weights_only=False)
                    mask_payload = torch.load(depth_path(mask_root, clip_id), map_location="cpu", weights_only=False)
                    mask = mask_payload.get("latent_validity")
                    if (source_payload.get("schema") != DEPTH_SCHEMA
                            or source_payload.get("cache_contract_hash") != depth_contract
                            or source_payload.get("clip_id") != clip_id
                            or not _finite_bf16_latent(source_payload.get("source_latent"))
                            or not _finite_bf16_latent(source_payload.get("target_latent"))):
                        raise ValueError(f"Invalid LingBot depth payload: {clip_id}")
                    if (normal_payload.get("schema") != LINGBOT_NORMAL_SCHEMA
                            or normal_payload.get("cache_contract_hash") != normal_contract
                            or normal_payload.get("clip_id") != clip_id
                            or int(normal_payload.get("task_id", -1)) != TASK_TO_ID["normal"]
                            or not _finite_bf16_latent(normal_payload.get("target_latent"))):
                        raise ValueError(f"Invalid LingBot normal payload: {clip_id}")
                    if (mask_payload.get("schema") != LINGBOT_MASK_SCHEMA
                            or mask_payload.get("cache_contract_hash") != mask_contract
                            or mask_payload.get("clip_id") != clip_id
                            or not isinstance(mask, torch.Tensor) or tuple(mask.shape) != MASK_SHAPE
                            or mask.dtype != torch.float16 or not torch.isfinite(mask).all().item()
                            or float(mask.min()) < 0.0 or float(mask.max()) > 1.0):
                        raise ValueError(f"Invalid LingBot mask payload: {clip_id}")
                    local_count += 3
        except Exception as error:
            local_error = f"rank={rank} {type(error).__name__}: {error}"
        errors: list[str | None] = [None] * world_size
        dist.all_gather_object(errors, local_error)
        failures = [error for error in errors if error is not None]
        if failures:
            raise RuntimeError("Dense6 payload audit failed: " + "; ".join(failures))
        count_device = (
            torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            if dist.get_backend() == "nccl" else torch.device("cpu")
        )
        count = torch.tensor(local_count, device=count_device)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        expected = self.num_clips * len(TASK_TO_ID) + len(self.lingbot_clip_ids) * 3
        if int(count.item()) != expected:
            raise RuntimeError(f"Payload audit count mismatch: {int(count.item())} != {expected}")

    def __len__(self) -> int:
        return self.num_clips * len(TASK_TO_ID) + len(self.lingbot_clip_ids) * 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        base_count = self.num_clips * len(TASK_TO_ID)
        if index >= base_count:
            if self.lingbot_roots is None or self.lingbot_contracts is None:
                raise IndexError(index)
            auxiliary_index = index - base_count
            task_id = TASK_TO_ID["depth"] if auxiliary_index < len(self.lingbot_clip_ids) else TASK_TO_ID["normal"]
            clip_index = auxiliary_index % len(self.lingbot_clip_ids)
            clip_id = self.lingbot_clip_ids[clip_index]
            depth_root, normal_root, mask_root = self.lingbot_roots
            depth_contract, normal_contract, mask_contract = self.lingbot_contracts
            source_payload = torch.load(depth_path(depth_root, clip_id), map_location="cpu", weights_only=False)
            if source_payload.get("cache_contract_hash") != depth_contract:
                raise ValueError(f"Invalid LingBot depth contract: {clip_id}")
            source = source_payload["source_latent"]
            if task_id == TASK_TO_ID["depth"]:
                target = source_payload["target_latent"]
                prompt = "output: depth"
            else:
                payload = torch.load(depth_path(normal_root, clip_id), map_location="cpu", weights_only=False)
                if payload.get("cache_contract_hash") != normal_contract:
                    raise ValueError(f"Invalid LingBot normal contract: {clip_id}")
                target, prompt = payload["target_latent"], payload["prompt"]
            mask_payload = torch.load(depth_path(mask_root, clip_id), map_location="cpu", weights_only=False)
            if mask_payload.get("cache_contract_hash") != mask_contract:
                raise ValueError(f"Invalid LingBot mask contract: {clip_id}")
            loss_mask = mask_payload["latent_validity"]
            return source, target, task_id, self.prompt_to_id[prompt], loss_mask
        clip_index, task_id = divmod(index, len(TASK_TO_ID))
        clip_id = self.clip_ids[clip_index]
        source_payload = torch.load(depth_path(self.depth_root, clip_id), map_location="cpu", weights_only=False)
        if (source_payload.get("schema") != self.depth_schema
                or (self.depth_contract_hash is not None
                    and source_payload.get("cache_contract_hash") != self.depth_contract_hash)):
            raise ValueError(f"Invalid depth cache payload: {clip_id}")
        source = source_payload["source_latent"]
        if task_id == TASK_TO_ID["depth"]:
            target = source_payload["target_latent"]
            prompt = "output: depth"
        else:
            task = ID_TO_TASK[task_id]
            task_root, task_schema, _, contract_hash = self.target_contract(task)
            payload = torch.load(
                target_path(task_root, task, clip_id), map_location="cpu", weights_only=False
            )
            if (
                payload.get("schema") != task_schema
                or payload.get("cache_contract_hash") != contract_hash
                or payload.get("clip_id") != clip_id
                or int(payload.get("task_id", -1)) != task_id
            ):
                raise ValueError(f"Invalid {task} target cache payload: {clip_id}")
            target = payload["target_latent"]
            prompt = payload["prompt"]
        if not _finite_bf16_latent(source) or not _finite_bf16_latent(target):
            raise ValueError(f"Invalid cached latent shape for {clip_id}/{ID_TO_TASK[task_id]}")
        if prompt not in self.prompt_to_id:
            raise ValueError(f"Unknown cached prompt {prompt!r}")
        return source, target, task_id, self.prompt_to_id[prompt], torch.ones(MASK_SHAPE, dtype=torch.float16)


class Dense6LoRAModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_dir: str,
        lora_rank: int,
        target_modules: str,
        router_hidden_dim: int,
        router_mode: str,
        router_identity_weight: float,
    ) -> None:
        super().__init__()
        model_dir = Path(model_dir).resolve()
        shards = sorted(model_dir.glob("diffusion_pytorch_model-*.safetensors"))
        expected_names = [f"diffusion_pytorch_model-0000{i}-of-00003.safetensors" for i in range(1, 4)]
        if [path.name for path in shards] != expected_names:
            raise ValueError(f"Expected exactly the three TI2V-5B DiT shards, got {[p.name for p in shards]}")
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[
                ModelConfig(path=[str(path) for path in shards]),
                ModelConfig(path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth")),
            ],
            tokenizer_config=ModelConfig(path=str(model_dir / "google/umt5-xxl")),
        )
        dit = self.pipe.dit
        model_contract = {
            "in_dim": dit.in_dim,
            "out_dim": dit.head.head.out_features // int(np.prod(dit.patch_size)),
            "num_blocks": len(dit.blocks),
            "patch_size": tuple(dit.patch_size),
            "seperated_timestep": dit.seperated_timestep,
            "fuse_vae_embedding_in_latents": dit.fuse_vae_embedding_in_latents,
        }
        expected_contract = {
            "in_dim": 48, "out_dim": 48, "num_blocks": 30,
            "patch_size": (1, 2, 2), "seperated_timestep": True,
            "fuse_vae_embedding_in_latents": True,
        }
        if model_contract != expected_contract:
            raise ValueError(f"Wrong Wan2.2 TI2V-5B model contract: {model_contract}")
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=None,
            lora_base_model="dit",
            lora_target_modules=target_modules,
            lora_rank=lora_rank,
            task="sft",
        )
        # Keep the adapter parameters and AdamW moments in FP32. PEFT's LoRA
        # layer casts its input to the adapter weight dtype and casts the
        # residual back, so the frozen BF16 Wan weights remain unchanged.
        for name, parameter in self.pipe.dit.named_parameters():
            if parameter.requires_grad and ".lora_" in name:
                parameter.data = parameter.data.float()
        sites = discover_lora_sites(self.pipe.dit)
        if len(sites) != 300:
            raise ValueError(f"Expected exactly 300 Wan2.2 LoRA sites, got {len(sites)}")
        if any(lora_a.out_features != lora_rank for _, lora_a in sites):
            raise ValueError("LoRA-A rank differs from the requested rank")
        self.task_router = TaskHyperRouter(
            [name for name, _ in sites],
            rank=lora_rank,
            hidden_dim=router_hidden_dim,
        )
        self.task_router.bind(self.pipe.dit)
        self.router_mode = router_mode
        self.router_identity_weight = float(router_identity_weight)
        if self.router_mode == "identity":
            self.task_router.requires_grad_(False)
        self.context_table: torch.Tensor | None = None
        self.provenance_metadata: dict[str, str] | None = None

    @torch.no_grad()
    def prepare_condition_tables(self, prompt_texts: list[str], device: torch.device) -> None:
        self.pipe.device = device
        self.pipe.text_encoder.to(device)
        encoder = WanVideoUnit_PromptEmbedder()
        contexts = []
        for prompt in prompt_texts:
            contexts.append(encoder.encode_prompt(self.pipe, prompt).detach().cpu())
        task_features = []
        for task_id in range(len(TASK_TO_ID)):
            task_context = encoder.encode_prompt(self.pipe, TASK_INSTRUCTIONS[task_id])
            task_features.append(task_context[0, 0].detach().float().cpu())
        self.context_table = torch.cat(contexts, dim=0).to(device=device, dtype=torch.bfloat16)
        self.task_router.set_task_feature_table(torch.stack(task_features))
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder = None
        self.pipe.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def forward(
        self,
        source_latent: torch.Tensor,
        target_latent: torch.Tensor,
        task_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        gradient_checkpointing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        velocity, gates = self.predict_velocity(
            source_latent, task_ids, prompt_ids, gradient_checkpointing
        )
        identity_loss = (
            self.task_router.identity_regularization(gates)
            if self.router_mode == "learned"
            else source_latent.new_zeros((), dtype=torch.float32)
        )
        data_loss = masked_latent_mse(-velocity, target_latent, loss_mask)
        total_loss = data_loss + self.router_identity_weight * identity_loss
        return total_loss, data_loss.detach(), identity_loss.detach()

    def predict_velocity(
        self,
        source_latent: torch.Tensor,
        task_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        gradient_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.context_table is None:
            raise RuntimeError("Prompt context table has not been prepared")
        if self.router_mode == "learned":
            gates = self.task_router(task_ids)
        else:
            gates = self.task_router.activate_identity(task_ids)
        timestep = torch.zeros(source_latent.shape[0], device=source_latent.device, dtype=torch.bfloat16)
        velocity = self.pipe.model_fn(
            dit=self.pipe.dit,
            latents=source_latent,
            timestep=timestep,
            context=self.context_table[prompt_ids],
            use_gradient_checkpointing=gradient_checkpointing,
            use_gradient_checkpointing_offload=False,
            cfg_merge=False,
        )
        return velocity, gates


def save_checkpoint(
    module: Dense6LoRAModule,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
) -> None:
    summary = module.task_router.routing_summary()
    path = output_dir / f"step-{step}.safetensors"
    extra_metadata = {"router_freeze_step": str(args.router_freeze_step)}
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint).resolve()
        extra_metadata.update({
            "initialization_checkpoint": str(init_path),
            "initialization_checkpoint_sha256": sha256_file(init_path),
        })
    save_dense6_checkpoint(
        module,
        path,
        step=step,
        world_size=dist.get_world_size(),
        lora_rank=args.lora_rank,
        lora_targets=args.lora_target_modules,
        router_hidden_dim=args.router_hidden_dim,
        router_mode=args.router_mode,
        provenance_metadata=module.provenance_metadata or {},
        extra_metadata=extra_metadata,
    )
    print(f"Saved {path}; routing={summary}", flush=True)


def optimizer_param_groups(module: Dense6LoRAModule, args: argparse.Namespace) -> list[dict]:
    lora: list[torch.nn.Parameter] = []
    router_decay: list[torch.nn.Parameter] = []
    router_no_decay: list[torch.nn.Parameter] = []
    assigned: set[int] = set()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("pipe.dit.") and ".lora_" in name:
            lora.append(parameter)
        elif name.startswith("task_router."):
            # LayerNorm, all biases and the zero-init routing head should not be
            # pulled away from the identity by decoupled weight decay.
            if name.endswith(".bias") or ".hypernet.0." in name or ".layernets." in name:
                router_no_decay.append(parameter)
            else:
                router_decay.append(parameter)
        else:
            raise RuntimeError(f"Unexpected trainable parameter: {name}")
        assigned.add(id(parameter))
    expected = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
    if assigned != expected or not lora:
        raise RuntimeError("Optimizer parameter grouping is incomplete")
    groups = [{"params": lora, "lr": args.learning_rate, "weight_decay": args.weight_decay, "name": "lora"}]
    if router_decay:
        groups.append(
            {"params": router_decay, "lr": args.router_learning_rate,
             "weight_decay": args.router_weight_decay, "name": "router_decay"}
        )
    if router_no_decay:
        groups.append(
            {"params": router_no_decay, "lr": args.router_learning_rate,
             "weight_decay": 0.0, "name": "router_no_decay"}
        )
    return groups


def gradients_are_finite(module: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in module.parameters()
    )


def freeze_router_if_due(
    module: Dense6LoRAModule,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    freeze_step: int,
) -> bool:
    if freeze_step < 0 or step < freeze_step:
        return False
    module.task_router.requires_grad_(False)
    if optimizer is not None:
        for group in optimizer.param_groups:
            if str(group.get("name", "")).startswith("router"):
                group["lr"] = 0.0
    return True


def trainer_state_path(output_dir: Path) -> Path:
    return output_dir / "trainer_state_latest.pt"


def trainer_rng_path(output_dir: Path, rank: int) -> Path:
    return output_dir / f"trainer_rng_rank{rank}_latest.pt"


def save_trainer_state(
    output_dir: Path,
    rank: int,
    *,
    global_step: int,
    epoch: int,
    next_batch_in_epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    common_state = {
        "global_step": global_step,
        "epoch": epoch,
        "next_batch_in_epoch": next_batch_in_epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    rng_state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    if rank == 0:
        path = trainer_state_path(output_dir)
        temporary = path.with_suffix(f".tmp-p{os.getpid()}")
        torch.save(common_state, temporary)
        os.replace(temporary, path)
    rng_path = trainer_rng_path(output_dir, rank)
    rng_temporary = rng_path.with_suffix(f".tmp-r{rank}-p{os.getpid()}")
    torch.save(rng_state, rng_temporary)
    os.replace(rng_temporary, rng_path)


def load_trainer_state(
    resume_from: str,
    output_dir: Path,
    rank: int,
    module: Dense6LoRAModule,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int, int]:
    resume_path = Path(resume_from).resolve()
    if not resume_path.is_dir():
        raise ValueError("--resume_from must name the previous output directory")
    state_path = trainer_state_path(resume_path)
    rng_path = trainer_rng_path(resume_path, rank)
    checkpoint_path = resume_path / "step-PLACEHOLDER.safetensors"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
    global_step = int(state["global_step"])
    checkpoint_path = checkpoint_path.with_name(f"step-{global_step}.safetensors")
    load_dense6_checkpoint(
        module=module,
        path=checkpoint_path,
        expected_provenance=module.provenance_metadata or {},
        lora_rank=args.lora_rank,
        lora_targets=args.lora_target_modules,
        router_hidden_dim=args.router_hidden_dim,
        expected_router_mode=args.router_mode,
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(rng_state["python_rng"])
    np.random.set_state(rng_state["numpy_rng"])
    torch.set_rng_state(rng_state["torch_rng"])
    torch.cuda.set_rng_state(rng_state["cuda_rng"])
    return global_step, int(state["epoch"]), int(state["next_batch_in_epoch"])


def main() -> None:
    args = parse_args()
    if args.resume_from and args.init_checkpoint:
        raise ValueError("--resume_from and --init_checkpoint are mutually exclusive")
    if (args.warmup_steps < 0 or args.max_grad_norm <= 0 or args.router_identity_weight < 0
            or args.router_freeze_step < -1):
        raise ValueError("warmup_steps, max_grad_norm and router_identity_weight are invalid")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.distributed_backend == "gloo":
        # Explicit smoke/debug fallback for hosts whose NCCL/CXI plugin blocks.
        # Model math remains CUDA; no silent fallback is allowed for formal runs.
        dist.init_process_group("gloo")
    else:
        dist.init_process_group("nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 4 and not (world_size == 1 and args.single_gpu_smoke):
        raise ValueError(f"Dense6 formal training requires 4 GPUs, got {world_size}")
    if args.single_gpu_smoke and world_size != 1:
        raise ValueError("--single_gpu_smoke requires exactly one process")

    output_dir = Path(args.output_path).resolve()
    output_error: list[str | None] = [None]
    if rank == 0:
        try:
            if output_dir.exists() and any(output_dir.iterdir()) and not (args.resume_from or args.allow_existing_output):
                output_error[0] = f"Refusing to overwrite non-empty output directory: {output_dir}"
            elif args.resume_from and Path(args.resume_from).resolve() != output_dir:
                output_error[0] = "Resume must use the same directory as --output_path"
            elif args.resume_from:
                previous = json.loads((output_dir / "training_args.json").read_text(encoding="utf-8"))
                for key in (
                    "depth_cache_dir", "target_cache_dir", "extra_target_cache_dir",
                    "dataset_root", "prompt_dir", "model_dir",
                    "code_patch_path", "lingbot_depth_cache_dir", "lingbot_normal_cache_dir",
                    "lingbot_mask_cache_dir", "init_checkpoint", "init_code_patch_path",
                    "learning_rate", "weight_decay", "router_learning_rate", "router_weight_decay",
                    "router_identity_weight", "router_mode", "lora_rank", "router_hidden_dim",
                    "lora_target_modules", "seed", "warmup_steps", "max_grad_norm",
                    "router_freeze_step", "single_gpu_smoke",
                    "distributed_backend", "allow_legacy_v1_cache",
                ):
                    if previous.get(key) != getattr(args, key):
                        output_error[0] = f"Resume argument mismatch for {key}"
                        break
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            output_error[0] = f"{type(error).__name__}: {error}"
    broadcast_object_list(output_error, src=0)
    if output_error[0]:
        raise FileExistsError(output_error[0])

    validate_cache_contract(args, rank, world_size)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    dataset = CachedDense6Dataset(
        args.depth_cache_dir, args.target_cache_dir, args.extra_target_cache_dir,
        allow_legacy_v1_cache=args.allow_legacy_v1_cache,
        lingbot_depth_cache_dir=args.lingbot_depth_cache_dir,
        lingbot_normal_cache_dir=args.lingbot_normal_cache_dir,
        lingbot_mask_cache_dir=args.lingbot_mask_cache_dir,
    )
    dataset.validate_all_payloads(rank, world_size)
    sampler = (
        ProportionalTaskPoolDistributedSampler(
            dataset.task_pools, num_replicas=world_size, rank=rank, seed=args.seed
        )
        if dataset.lingbot_clip_ids else
        SixTaskStratifiedDistributedSampler(
            dataset.num_clips, num_replicas=world_size, rank=rank, seed=args.seed
        )
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    module = Dense6LoRAModule(
        args.model_dir,
        args.lora_rank,
        args.lora_target_modules,
        args.router_hidden_dim,
        args.router_mode,
        args.router_identity_weight,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(dataset.prompt_texts, device)
    provenance_holder: list[dict[str, str] | None] = [None]
    base_provenance_holder: list[dict[str, str] | None] = [None]
    if rank == 0:
        provenance_kwargs = dict(
            module=module,
            model_dir=args.model_dir,
            prompt_texts=dataset.prompt_texts,
            prompt_dir=args.prompt_dir,
            depth_cache_dir=args.depth_cache_dir,
            target_cache_dir=args.target_cache_dir,
            code_patch_path=args.code_patch_path,
            repo_dir=THIS_DIR.parents[3],
        )
        # An initialization checkpoint is checked against its original cache
        # semantics; a continuation may additionally bind the LingBot caches.
        init_provenance_kwargs = dict(provenance_kwargs)
        if args.init_code_patch_path:
            init_provenance_kwargs["code_patch_path"] = args.init_code_patch_path
        base_auxiliary_dirs = {"albedo_delayed_flow": args.extra_target_cache_dir}
        base_provenance_holder[0] = build_provenance_metadata(
            **init_provenance_kwargs, auxiliary_cache_dirs=base_auxiliary_dirs
        )
        auxiliary_dirs = dict(base_auxiliary_dirs)
        if args.lingbot_depth_cache_dir:
            auxiliary_dirs.update({
                "lingbot_depth": args.lingbot_depth_cache_dir,
                "lingbot_normal": args.lingbot_normal_cache_dir,
                "lingbot_validity": args.lingbot_mask_cache_dir,
            })
        provenance_holder[0] = build_provenance_metadata(
            **provenance_kwargs, auxiliary_cache_dirs=auxiliary_dirs
        )
    broadcast_object_list(provenance_holder, src=0)
    broadcast_object_list(base_provenance_holder, src=0)
    if not isinstance(provenance_holder[0], dict):
        raise RuntimeError("Runtime provenance broadcast failed")
    module.provenance_metadata = provenance_holder[0]
    if args.init_checkpoint:
        if not isinstance(base_provenance_holder[0], dict):
            raise RuntimeError("Base provenance broadcast failed")
        load_dense6_checkpoint(
            module, args.init_checkpoint,
            expected_provenance=base_provenance_holder[0],
            lora_rank=args.lora_rank,
            lora_targets=args.lora_target_modules,
            router_hidden_dim=args.router_hidden_dim,
            expected_router_mode=args.router_mode,
        )
    module.pipe.dit.to(device).train()
    freeze_router_if_due(module, None, 0, args.router_freeze_step)
    bad_lora_dtype = [
        name for name, parameter in module.named_parameters()
        if parameter.requires_grad and ".lora_" in name and parameter.dtype != torch.float32
    ]
    if bad_lora_dtype:
        raise RuntimeError(f"LoRA parameters must remain FP32: {bad_lora_dtype[:10]}")
    bad_trainable = [
        name for name, parameter in module.named_parameters()
        if parameter.requires_grad
        and not (name.startswith("task_router.") or (name.startswith("pipe.dit.") and ".lora_" in name))
    ]
    if bad_trainable:
        raise RuntimeError(f"Unexpected trainable parameters: {bad_trainable[:20]}")
    trainable_count = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    router_count = sum(parameter.numel() for parameter in module.task_router.parameters())

    if rank == 0:
        config = vars(args) | {
            "task_to_id": TASK_TO_ID,
            "task_instructions": TASK_INSTRUCTIONS,
            "prompt_texts": dataset.prompt_texts,
            "num_clips": dataset.num_clips,
            "task_samples": len(dataset),
            "task_pool_sizes": {ID_TO_TASK[task]: len(pool) for task, pool in dataset.task_pools.items()},
            "steps_per_epoch": len(dataloader),
            "lora_sites": list(module.task_router.site_names),
            "trainable_count": trainable_count,
            "router_count": router_count,
            "provenance": module.provenance_metadata,
            "sampling": (
                "proportional task-pool sampling; every sample once plus at most three DDP padding samples"
                if dataset.lingbot_clip_ids else
                ("every three global steps contain each of task IDs 0..5 exactly twice"
                 if world_size == 4 else "single-GPU smoke cycles task IDs 0..5")
            ),
        }
        config_path = output_dir / "training_args.json"
        if not args.resume_from:
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"clips={dataset.num_clips} task_samples={len(dataset)} world={world_size} "
            f"trainable={trainable_count:,} router={router_count:,} sites={len(module.task_router.site_names)}",
            flush=True,
        )
    dist.barrier()

    ddp = DistributedDataParallel(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=args.router_freeze_step >= 0,
    )
    optimizer = torch.optim.AdamW(optimizer_param_groups(module, args))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda current_step: (
            1.0 if args.warmup_steps == 0 else min(1.0, float(current_step + 1) / args.warmup_steps)
        ),
    )
    step = epoch = next_batch_in_epoch = 0
    if args.resume_from:
        step, epoch, next_batch_in_epoch = load_trainer_state(
            args.resume_from, output_dir, rank, module, args, optimizer, scheduler
        )
        if step >= args.max_train_steps:
            raise ValueError(f"Resume step {step} already reached max_train_steps={args.max_train_steps}")
        print(f"rank={rank} resumed step={step} epoch={epoch} batch={next_batch_in_epoch}", flush=True)
    freeze_router_if_due(module, optimizer, step, args.router_freeze_step)
    loss_path = output_dir / "loss.csv"
    gate_path = output_dir / "gate_metrics.jsonl"
    if rank == 0 and not args.resume_from:
        with loss_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [
                    "step", "pre_update_total_loss", "pre_update_data_loss", "identity_loss",
                    "task_id", "task", "grad_norm", "seconds", "max_memory_gib",
                    "lora_lr", "router_lr", "mask_valid_fraction",
                ]
            )
        gate_path.write_text("", encoding="utf-8")

    start = time.time()
    resumed_step = step
    smoke_seen_tasks: set[int] = set()
    stratified_window_tasks: list[int] = []
    while step < args.max_train_steps:
        sampler.set_epoch(epoch)
        for batch_position, (source, target, task_ids, prompt_ids, loss_mask) in enumerate(dataloader):
            if batch_position < next_batch_in_epoch:
                continue
            source = source.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            target = target.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            task_ids = task_ids.to(device=device, dtype=torch.long, non_blocking=True)
            prompt_ids = prompt_ids.to(device=device, dtype=torch.long, non_blocking=True)
            loss_mask = loss_mask.to(device=device, dtype=torch.float32, non_blocking=True)
            freeze_router_if_due(module, optimizer, step, args.router_freeze_step)
            optimizer.zero_grad(set_to_none=True)
            loss, data_loss, identity_loss = module.task_router.call_with_error_cleanup(
                lambda: ddp(source, target, task_ids, prompt_ids, loss_mask, args.gradient_checkpointing)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"rank={rank} non-finite loss at step {step + 1}")
            loss.backward()
            finite = torch.tensor(int(gradients_are_finite(module)), device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite.item()):
                raise FloatingPointError(f"Non-finite gradient before step {step + 1}")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in module.parameters() if parameter.requires_grad],
                args.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            freeze_router_if_due(module, optimizer, step + 1, args.router_freeze_step)
            module.task_router.clear()
            step += 1

            # Log one row per rank/task so routing behavior remains auditable.
            record = torch.tensor(
                [
                    float(task_ids.item()), float(loss.detach()), float(data_loss),
                    float(identity_loss), float(grad_norm), float(loss_mask.mean()),
                ],
                device=device,
                dtype=torch.float32,
            )
            gathered = [torch.empty_like(record) for _ in range(world_size)]
            dist.all_gather(gathered, record)
            if rank == 0:
                gathered_task_ids = {int(item[0]) for item in gathered}
                smoke_seen_tasks.update(gathered_task_ids)
                if world_size == 4 and not dataset.lingbot_clip_ids:
                    if step % 3 == 1:
                        stratified_window_tasks.clear()
                    stratified_window_tasks.extend(int(item[0]) for item in gathered)
                    if step % 3 == 0 and len(stratified_window_tasks) == 12:
                        counts = {
                            task_id: stratified_window_tasks.count(task_id)
                            for task_id in TASK_TO_ID.values()
                        }
                        if set(stratified_window_tasks) != set(TASK_TO_ID.values()) or any(
                            count != 2 for count in counts.values()
                        ):
                            raise RuntimeError(
                                f"Three-step Dense6 stratification failed at step {step}: {counts}"
                            )
                elapsed = time.time() - start
                global_loss = sum(float(item[1]) for item in gathered) / world_size
                memory = torch.cuda.max_memory_allocated(device) / 1024**3
                with loss_path.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    for item in gathered:
                        task_id = int(item[0])
                        writer.writerow(
                            [
                                step, float(item[1]), float(item[2]), float(item[3]), task_id,
                                ID_TO_TASK[task_id], float(item[4]), elapsed, memory,
                                optimizer.param_groups[0]["lr"],
                                next(
                                    (group["lr"] for group in optimizer.param_groups if group.get("name", "").startswith("router")),
                                    0.0,
                                ),
                                float(item[5]),
                            ]
                        )
                if step == 1 or step % 10 == 0:
                    routing = module.task_router.routing_summary()
                    with gate_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"step": step, **routing}, sort_keys=True) + "\n")
                    print(
                        f"step={step}/{args.max_train_steps} loss={global_loss:.6f} "
                        f"grad_norm={sum(float(item[4]) for item in gathered)/world_size:.4f} "
                        f"sec_per_step={elapsed/max(1, step-resumed_step):.3f} max_mem_gib={memory:.3f}",
                        flush=True,
                    )

            if step % args.save_steps == 0 or step == args.max_train_steps:
                dist.barrier()
                if rank == 0:
                    save_checkpoint(module, output_dir, step, args)
                dist.barrier()
                saved_epoch = epoch
                saved_next_batch = batch_position + 1
                if saved_next_batch >= len(dataloader):
                    saved_epoch += 1
                    saved_next_batch = 0
                save_trainer_state(
                    output_dir,
                    rank,
                    global_step=step,
                    epoch=saved_epoch,
                    next_batch_in_epoch=saved_next_batch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                dist.barrier()
            if step >= args.max_train_steps:
                break
        epoch += 1
        next_batch_in_epoch = 0
    if rank == 0 and args.single_gpu_smoke and args.max_train_steps >= 6:
        if smoke_seen_tasks != set(TASK_TO_ID.values()):
            raise RuntimeError(f"Single-GPU smoke did not cover every task: {smoke_seen_tasks}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
