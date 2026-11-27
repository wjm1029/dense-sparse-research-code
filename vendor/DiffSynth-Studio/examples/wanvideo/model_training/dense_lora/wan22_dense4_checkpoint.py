"""Save and restore unfused PEFT LoRA plus Dense4 task-routing semantics."""

from __future__ import annotations

import json
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from wan22_cache_integrity import sha256_file, sha256_json, sha256_lines, sha256_named_files
from wan22_task_router import TASK_INSTRUCTIONS, TASK_TO_ID

if TYPE_CHECKING:
    from train_wan22_dense4_ddp import Dense4LoRAModule


CHECKPOINT_SCHEMA = "wan22_ti2v5b_dense4_unfused_hyperlora_v2"
REQUIRED_PROVENANCE_KEYS = {
    "base_model_identity",
    "output_prompt_texts",
    "output_prompt_texts_sha256",
    "prompt_manifest_sha256",
    "task_feature_table_sha256",
    "depth_cache_manifest_sha256",
    "depth_cache_contract_hash",
    "target_cache_manifest_sha256",
    "target_cache_contract_hash",
    "diffsynth_git_head",
    "runtime_source_patch_sha256",
}
OPTIONAL_PROVENANCE_KEYS = {"auxiliary_cache_identities"}


def _validate_provenance_keys(metadata: dict[str, str]) -> None:
    keys = set(metadata)
    if not REQUIRED_PROVENANCE_KEYS.issubset(keys) or keys - REQUIRED_PROVENANCE_KEYS - OPTIONAL_PROVENANCE_KEYS:
        raise ValueError("Checkpoint provenance is incomplete or contains unknown fields")


@lru_cache(maxsize=4)
def _base_model_identity_cached(model_dir: str) -> dict[str, object]:
    root = Path(model_dir)
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    config_path = root / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    expected_shards = [f"diffusion_pytorch_model-0000{i}-of-00003.safetensors" for i in range(1, 4)]
    if shards != expected_shards:
        raise ValueError(f"Expected exactly the three Wan2.2 TI2V-5B shards, got {shards}")
    tokenizer_root = root / "google" / "umt5-xxl"
    tokenizer_files = [path for path in tokenizer_root.rglob("*") if path.is_file()]
    if not tokenizer_files:
        raise FileNotFoundError(f"Missing UMT5 tokenizer files: {tokenizer_root}")
    return {
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "shards": [
            {"name": name, "size": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
            for name in shards
        ],
        "umt5_sha256": sha256_file(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        "tokenizer_sha256": sha256_named_files(tokenizer_files, tokenizer_root),
    }


def base_model_identity(model_dir: str | Path) -> dict[str, object]:
    return _base_model_identity_cached(str(Path(model_dir).resolve()))


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def build_provenance_metadata(
    module: "Dense4LoRAModule",
    *,
    model_dir: str | Path,
    prompt_texts: list[str],
    prompt_dir: str | Path,
    depth_cache_dir: str | Path,
    target_cache_dir: str | Path,
    code_patch_path: str | Path,
    repo_dir: str | Path,
    auxiliary_cache_dirs: dict[str, str | Path] | None = None,
) -> dict[str, str]:
    patch = Path(code_patch_path).resolve()
    if not patch.is_file():
        raise FileNotFoundError(f"Missing immutable runtime source patch: {patch}")
    depth_manifest_path = Path(depth_cache_dir).resolve() / "manifest.json"
    target_manifest_path = Path(target_cache_dir).resolve() / "manifest.json"
    depth_manifest = json.loads(depth_manifest_path.read_text(encoding="utf-8"))
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(repo_dir).resolve(), check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    prompt_json = json.dumps(prompt_texts, ensure_ascii=False, separators=(",", ":"))
    model_json = json.dumps(base_model_identity(model_dir), sort_keys=True, separators=(",", ":"))
    result = {
        "base_model_identity": model_json,
        "output_prompt_texts": prompt_json,
        "output_prompt_texts_sha256": sha256_json(prompt_texts),
        "prompt_manifest_sha256": sha256_named_files(
            list(Path(prompt_dir).resolve().glob("*.jsonl")), Path(prompt_dir).resolve()
        ),
        "task_feature_table_sha256": tensor_sha256(module.task_router.task_feature_table),
        "depth_cache_manifest_sha256": sha256_file(depth_manifest_path),
        "depth_cache_contract_hash": str(
            depth_manifest.get("cache_contract_hash")
            or f"legacy-v1-manifest:{sha256_file(depth_manifest_path)}"
        ),
        "target_cache_manifest_sha256": sha256_file(target_manifest_path),
        "target_cache_contract_hash": str(target_manifest["cache_contract_hash"]),
        "diffsynth_git_head": git_head,
        "runtime_source_patch_sha256": sha256_file(patch),
    }
    if auxiliary_cache_dirs:
        identities = {}
        for name, directory in sorted(auxiliary_cache_dirs.items()):
            root = Path(directory).resolve()
            manifest_path = root / "manifest.json"
            complete_path = root / "CACHE_COMPLETE"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            manifest_sha = sha256_file(manifest_path)
            if complete.get("manifest_sha256") != manifest_sha:
                raise ValueError(f"Auxiliary cache manifest hash mismatch: {name}")
            if complete.get("cache_contract_hash") != manifest.get("cache_contract_hash"):
                raise ValueError(f"Auxiliary cache contract mismatch: {name}")
            identities[name] = {
                "path": str(root),
                "schema": manifest.get("schema"),
                "manifest_sha256": manifest_sha,
                "cache_contract_hash": manifest.get("cache_contract_hash"),
                "num_samples": complete.get("num_samples"),
            }
        result["auxiliary_cache_identities"] = json.dumps(
            identities, sort_keys=True, separators=(",", ":")
        )
    return result


def site_names_hash(site_names: tuple[str, ...] | list[str]) -> str:
    return sha256_lines(site_names)


def checkpoint_metadata(path: str | Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        return dict(stream.metadata() or {})


def save_dense4_checkpoint(
    module: "Dense4LoRAModule",
    path: str | Path,
    *,
    step: int,
    world_size: int,
    lora_rank: int,
    lora_targets: str,
    router_hidden_dim: int,
    router_mode: str,
    provenance_metadata: dict[str, str],
    extra_metadata: dict[str, str] | None = None,
) -> None:
    _validate_provenance_keys(provenance_metadata)
    trainable_names = set(module.trainable_param_names())
    state: dict[str, torch.Tensor] = {}
    for name, tensor in module.state_dict().items():
        if name in trainable_names and name.startswith("pipe.dit.") and ".lora_" in name:
            state[name.removeprefix("pipe.dit.")] = tensor.detach().cpu().contiguous()
        elif name.startswith("task_router."):
            state[name] = tensor.detach().cpu().contiguous()

    sites = list(module.task_router.site_names)
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "tasks": json.dumps(TASK_TO_ID, sort_keys=True),
        "task_instructions": json.dumps(TASK_INSTRUCTIONS, sort_keys=True),
        "method": "genception_t0_shared_lora_unicontrol_inspired_rank_hypernet",
        "base_model": "Wan2.2-TI2V-5B",
        "step": str(step),
        "world_size": str(world_size),
        "lora_rank": str(lora_rank),
        "lora_targets": lora_targets,
        "router_hidden_dim": str(router_hidden_dim),
        "router_mode": router_mode,
        "lora_site_names": json.dumps(sites, separators=(",", ":")),
        "lora_site_names_hash": site_names_hash(sites),
        "routing_summary": json.dumps(module.task_router.routing_summary(), sort_keys=True),
        "min_rgb_mean": "40.0",
    }
    metadata.update(provenance_metadata)
    if extra_metadata:
        if set(extra_metadata) & set(metadata):
            raise ValueError("Extra checkpoint metadata cannot override semantic provenance")
        metadata.update(extra_metadata)
    save_file(state, str(path), metadata=metadata)


@torch.no_grad()
def load_dense4_checkpoint(
    module: "Dense4LoRAModule",
    path: str | Path,
    *,
    expected_provenance: dict[str, str],
    lora_rank: int,
    lora_targets: str,
    router_hidden_dim: int,
    expected_router_mode: str = "learned",
) -> dict[str, str]:
    """Strictly load a Dense4 checkpoint without fusing away task routing."""
    path = Path(path).resolve()
    _validate_provenance_keys(expected_provenance)
    metadata = checkpoint_metadata(path)
    expected_metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "tasks": json.dumps(TASK_TO_ID, sort_keys=True),
        "task_instructions": json.dumps(TASK_INSTRUCTIONS, sort_keys=True),
        "base_model": "Wan2.2-TI2V-5B",
        "lora_rank": str(lora_rank),
        "lora_targets": lora_targets,
        "router_hidden_dim": str(router_hidden_dim),
        "router_mode": expected_router_mode,
    }
    expected_metadata.update(expected_provenance)
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(f"Checkpoint metadata mismatch for {key}: {actual!r} != {expected!r}")

    checkpoint_sites = tuple(json.loads(metadata["lora_site_names"]))
    current_sites = tuple(module.task_router.site_names)
    if checkpoint_sites != current_sites:
        raise ValueError("Checkpoint LoRA site ordering differs from the current PEFT model")
    if metadata.get("lora_site_names_hash") != site_names_hash(current_sites):
        raise ValueError("Checkpoint LoRA site-name hash is invalid")

    tensors = load_file(str(path), device="cpu")
    if metadata.get("output_prompt_texts_sha256") != sha256_json(
        json.loads(metadata["output_prompt_texts"])
    ):
        raise ValueError("Checkpoint output prompt text hash is internally inconsistent")
    saved_feature_name = "task_router.task_feature_table"
    if saved_feature_name not in tensors:
        raise ValueError("Checkpoint is missing the task feature table")
    if tensor_sha256(tensors[saved_feature_name]) != metadata.get("task_feature_table_sha256"):
        raise ValueError("Checkpoint task feature tensor does not match its semantic hash")
    module_state = module.state_dict()
    expected_lora = {
        name.removeprefix("pipe.dit."): name
        for name in module.trainable_param_names()
        if name.startswith("pipe.dit.") and ".lora_" in name
    }
    saved_lora = {name for name in tensors if not name.startswith("task_router.")}
    if saved_lora != set(expected_lora):
        missing = sorted(set(expected_lora) - saved_lora)
        extra = sorted(saved_lora - set(expected_lora))
        raise ValueError(f"LoRA tensor set mismatch; missing={missing[:5]}, extra={extra[:5]}")

    expected_router = {name for name in module_state if name.startswith("task_router.")}
    saved_router = {name for name in tensors if name.startswith("task_router.")}
    if saved_router != expected_router:
        missing = sorted(expected_router - saved_router)
        extra = sorted(saved_router - expected_router)
        raise ValueError(f"Router tensor set mismatch; missing={missing}, extra={extra}")

    for saved_name, module_name in expected_lora.items():
        source, destination = tensors[saved_name], module_state[module_name]
        if source.shape != destination.shape:
            raise ValueError(f"Shape mismatch for {saved_name}: {source.shape} != {destination.shape}")
        destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
    for name in expected_router:
        source, destination = tensors[name], module_state[name]
        if source.shape != destination.shape:
            raise ValueError(f"Shape mismatch for {name}: {source.shape} != {destination.shape}")
        destination.copy_(source.to(device=destination.device, dtype=destination.dtype))

    if not bool(module.task_router.features_ready.item()):
        raise RuntimeError("Checkpoint did not restore initialized task features")
    if not module.task_router._hook_handles:
        module.task_router.bind(module.pipe.dit)
    return metadata
