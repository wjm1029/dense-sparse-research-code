"""Save and restore unfused PEFT LoRA plus Dense4 task-routing semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from cache_integrity import sha256_file, sha256_lines
from task_router import TASK_INSTRUCTIONS, TASK_TO_ID

if TYPE_CHECKING:
    from train_dense4_ddp import Dense4LoRAModule


CHECKPOINT_SCHEMA = "wan21_dense4_unfused_hyperlora_v2"


def base_model_checkpoint(model_dir: str | Path) -> Path:
    return Path(model_dir).resolve() / "diffusion_pytorch_model.safetensors"


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
    model_dir: str | Path,
    extra_metadata: dict[str, str] | None = None,
) -> None:
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
        "base_model": "Wan2.1-T2V-1.3B",
        "base_model_sha256": sha256_file(base_model_checkpoint(model_dir)),
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
    if extra_metadata:
        metadata.update(extra_metadata)
    save_file(state, str(path), metadata=metadata)


@torch.no_grad()
def load_dense4_checkpoint(
    module: "Dense4LoRAModule",
    path: str | Path,
    *,
    model_dir: str | Path,
    lora_rank: int,
    lora_targets: str,
    router_hidden_dim: int,
    expected_router_mode: str = "learned",
) -> dict[str, str]:
    """Strictly load a Dense4 checkpoint without fusing away task routing."""
    path = Path(path).resolve()
    metadata = checkpoint_metadata(path)
    expected_metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "tasks": json.dumps(TASK_TO_ID, sort_keys=True),
        "task_instructions": json.dumps(TASK_INSTRUCTIONS, sort_keys=True),
        "base_model_sha256": sha256_file(base_model_checkpoint(model_dir)),
        "lora_rank": str(lora_rank),
        "lora_targets": lora_targets,
        "router_hidden_dim": str(router_hidden_dim),
        "router_mode": expected_router_mode,
    }
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
