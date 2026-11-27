"""Model-level equivalence and speed check for the pre-RoPE fixed KV pool."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from einops import rearrange


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import infer_wan22_ar_tf_long_video as long_infer
import infer_wan22_dense4 as common
import train_wan22_ar_tf_ddp as ar_train
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from wan22_ar_teacher_forcing import combine_rgb_and_history, install_ar_history_adapter
from wan22_dense4_checkpoint import (
    OPTIONAL_PROVENANCE_KEYS,
    REQUIRED_PROVENANCE_KEYS,
    checkpoint_metadata,
    load_dense4_checkpoint,
)
from wan22_opsd_bounded_causal_attention import install_opsd_bounded_causal_attention
from wan22_prerope_kv_pool import install_prerope_fixed_pool_attention


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_latent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_dir", default="/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B")
    parser.add_argument("--task", choices=("depth", "normal"), default="depth")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


class Local12Module(ar_train.dense4.Dense4LoRAModule):
    def __init__(self, *args, pool: bool, **kwargs):
        super().__init__(*args, **kwargs)
        installer = install_prerope_fixed_pool_attention if pool else install_opsd_bounded_causal_attention
        self.causal_controller = installer(
            self.pipe.dit, latent_shape=long_infer.LONG_LATENT_SHAPE,
            frames_per_chunk=3, local_attn_frames=12, sink_frames=3,
        )
        self.ar_controller = install_ar_history_adapter(
            self.pipe.dit, self.causal_controller, long_infer.LONG_AR_CONFIG
        )


@torch.no_grad()
def chunk_forward(controller, dit, x, timestep, context, frame_offset, caches):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(x.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    x = dit.patchify(x)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    temporal_start = frame_offset // dit.patch_size[0]
    freqs = torch.cat((
        dit.freqs[0][temporal_start:temporal_start + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ), dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    with controller.use_kv_cache(caches):
        for block in dit.blocks:
            x = block(x, context, t_mod, freqs)
    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))


@torch.no_grad()
def rollout(module, source, task_id, prompt_id):
    gates = module.task_router(task_id)
    timestep = torch.zeros(1, device=source.device, dtype=torch.bfloat16)
    context = module.context_table[prompt_id]
    caches = module.causal_controller.new_kv_cache()
    previous = torch.zeros_like(source[:, :, :3])
    outputs = []
    pointers_after_first = None
    started = time.perf_counter()
    for offset in range(0, 120, 3):
        ar_input = combine_rgb_and_history(source[:, :, offset:offset + 3], previous, long_infer.LONG_AR_CONFIG)
        prediction = -chunk_forward(
            module.causal_controller, module.pipe.dit, ar_input,
            timestep, context, offset, caches,
        )
        outputs.append(prediction)
        previous = prediction.detach()
        if offset == 0 and hasattr(caches[0], "data_ptrs"):
            pointers_after_first = tuple(cache.data_ptrs() for cache in caches)
    torch.cuda.synchronize(source.device)
    elapsed = time.perf_counter() - started
    pointers_final = (
        tuple(cache.data_ptrs() for cache in caches)
        if hasattr(caches[0], "data_ptrs") else None
    )
    cache_bytes = 0
    for cache in caches:
        for value in vars(cache).values():
            if isinstance(value, torch.Tensor):
                cache_bytes += value.numel() * value.element_size()
    module.task_router.clear()
    return torch.cat(outputs, dim=2).cpu(), {
        "elapsed_seconds": elapsed,
        "output_fps": 477.0 / elapsed,
        "cache_gib": cache_bytes / 2**30,
        "pool_storage_stable": pointers_after_first == pointers_final if pointers_final else None,
        "gate_mean": float(gates.mean()),
    }


def run_mode(args, metadata, source_cpu, pool):
    device = torch.device(args.device)
    prompt_texts = json.loads(metadata["output_prompt_texts"])
    module = Local12Module(
        args.model_dir, int(metadata["lora_rank"]), metadata["lora_targets"],
        int(metadata["router_hidden_dim"]), metadata["router_mode"], 0.0, pool=pool,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(prompt_texts, device)
    provenance = {
        key: metadata[key]
        for key in REQUIRED_PROVENANCE_KEYS | OPTIONAL_PROVENANCE_KEYS
        if key in metadata
    }
    module.pipe.dit.to(device).eval()
    load_dense4_checkpoint(
        module, args.checkpoint, expected_provenance=provenance,
        lora_rank=int(metadata["lora_rank"]), lora_targets=metadata["lora_targets"],
        router_hidden_dim=int(metadata["router_hidden_dim"]),
        expected_router_mode=metadata["router_mode"],
    )
    source = source_cpu.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    task_id = torch.tensor([common.TASK_TO_ID[args.task]], device=device)
    prompt_id = torch.tensor([
        common.prompt_id_for_task(prompt_texts, args.task, "person")
    ], device=device)
    torch.cuda.reset_peak_memory_stats(device)
    _, warmup = rollout(module, source, task_id, prompt_id)
    prediction, result = rollout(module, source, task_id, prompt_id)
    result["warmup_seconds"] = warmup["elapsed_seconds"]
    result["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    del module, source
    gc.collect()
    torch.cuda.empty_cache()
    return prediction, result


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    metadata = checkpoint_metadata(args.checkpoint)
    expected = {
        "causal_local_attn_frames": "12", "causal_sink_frames": "3",
        "opsd_attention_context": "upstream_local", "ar_frames_per_chunk": "3",
    }
    mismatch = {k: (metadata.get(k), v) for k, v in expected.items() if metadata.get(k) != v}
    if mismatch:
        raise ValueError(f"checkpoint is not local-12 OPSD: {mismatch}")
    payload = torch.load(args.source_latent, map_location="cpu", weights_only=False)
    source = payload["latent"]
    if tuple(source.shape) != long_infer.LONG_LATENT_SHAPE:
        raise ValueError("source latent shape mismatch")
    reference, reference_stats = run_mode(args, metadata, source, False)
    candidate, candidate_stats = run_mode(args, metadata, source, True)
    delta = (candidate.float() - reference.float()).abs()
    report = {
        "schema": "wan22_prerope_fixed_pool_validation_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_latent": str(Path(args.source_latent).resolve()),
        "task": args.task,
        "reference_post_rope_dynamic": reference_stats,
        "candidate_prerope_fixed_pool": candidate_stats,
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "speedup": reference_stats["elapsed_seconds"] / candidate_stats["elapsed_seconds"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
