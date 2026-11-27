"""Four-task output regression for the unfused Dense4 checkpoint loader."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from pathlib import Path

import torch

from dense4_checkpoint import checkpoint_metadata, load_dense4_checkpoint, save_dense4_checkpoint
from task_router import TASK_TO_ID
from train_dense4_ddp import CachedDense4Dataset, Dense4LoRAModule, depth_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--target_cache_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def build_loaded_module(args: argparse.Namespace, metadata: dict[str, str], dataset: CachedDense4Dataset):
    module = Dense4LoRAModule(
        args.model_dir,
        int(metadata["lora_rank"]),
        metadata["lora_targets"],
        int(metadata["router_hidden_dim"]),
        metadata["router_mode"],
        0.0,
    )
    module.task_router.to(device="cuda", dtype=torch.float32)
    module.prepare_condition_tables(dataset.prompt_texts, torch.device("cuda"))
    module.pipe.dit.to("cuda").eval()
    load_dense4_checkpoint(
        module,
        args.checkpoint,
        model_dir=args.model_dir,
        lora_rank=int(metadata["lora_rank"]),
        lora_targets=metadata["lora_targets"],
        router_hidden_dim=int(metadata["router_hidden_dim"]),
        expected_router_mode=metadata["router_mode"],
    )
    return module


@torch.no_grad()
def collect_outputs(module: Dense4LoRAModule, dataset: CachedDense4Dataset, source: torch.Tensor):
    outputs = {}
    gates = {}
    for task, task_id in TASK_TO_ID.items():
        _, _, _, prompt_id = dataset[task_id]
        task_tensor = torch.tensor([task_id], device="cuda", dtype=torch.long)
        prompt_tensor = torch.tensor([prompt_id], device="cuda", dtype=torch.long)
        velocity, task_gates = module.predict_velocity(source, task_tensor, prompt_tensor)
        outputs[task] = velocity.detach().cpu()
        gates[task] = task_gates.detach().cpu()
        module.task_router.clear()
    return outputs, gates


def main() -> None:
    args = parse_args()
    original_checkpoint = str(Path(args.checkpoint).resolve())
    metadata = checkpoint_metadata(args.checkpoint)
    dataset = CachedDense4Dataset(args.depth_cache_dir, args.target_cache_dir)
    source_payload = torch.load(
        depth_path(Path(args.depth_cache_dir), dataset.clip_ids[0]),
        map_location="cpu",
        weights_only=False,
    )
    source = source_payload["source_latent"].unsqueeze(0).to("cuda", dtype=torch.bfloat16)

    module = build_loaded_module(args, metadata, dataset)
    reference_outputs, reference_gates = collect_outputs(module, dataset, source)
    with tempfile.TemporaryDirectory(prefix="dense4-roundtrip-") as temporary_dir:
        roundtrip = Path(temporary_dir) / "roundtrip.safetensors"
        save_dense4_checkpoint(
            module,
            roundtrip,
            step=int(metadata["step"]),
            world_size=1,
            lora_rank=int(metadata["lora_rank"]),
            lora_targets=metadata["lora_targets"],
            router_hidden_dim=int(metadata["router_hidden_dim"]),
            router_mode=metadata["router_mode"],
            model_dir=args.model_dir,
        )
        del module
        gc.collect()
        torch.cuda.empty_cache()
        args.checkpoint = str(roundtrip)
        reloaded = build_loaded_module(args, metadata, dataset)
        reloaded_outputs, reloaded_gates = collect_outputs(reloaded, dataset, source)

    result = {"checkpoint": original_checkpoint, "tasks": {}}
    for task in TASK_TO_ID:
        output_diff = (reference_outputs[task].float() - reloaded_outputs[task].float()).abs()
        gate_diff = (reference_gates[task].float() - reloaded_gates[task].float()).abs()
        result["tasks"][task] = {
            "output_max_abs_diff": float(output_diff.max()),
            "gate_max_abs_diff": float(gate_diff.max()),
        }
        if not torch.equal(reference_outputs[task], reloaded_outputs[task]):
            raise AssertionError(f"Reloaded {task} output is not bitwise identical")
        if not torch.equal(reference_gates[task], reloaded_gates[task]):
            raise AssertionError(f"Reloaded {task} gates are not bitwise identical")
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
