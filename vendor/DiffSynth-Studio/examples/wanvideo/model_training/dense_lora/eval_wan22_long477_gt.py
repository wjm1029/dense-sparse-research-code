"""Paired latent-space GT evaluation for 477-frame Wan2.2 AR checkpoints.

The cached synthetic clips are training-pool examples, so this script labels its
output as in-domain rather than claiming held-out generalization. It performs
free-running AR inference and compares every predicted latent frame with the
cached VAE target, including first/last-quarter and temporal-delta diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import infer_wan22_ar_tf_long_video as long_infer
import infer_wan22_dense4 as common
import train_wan22_ar_tf_ddp as ar_train
from wan22_cache_integrity import sha256_file
from wan22_dense4_checkpoint import checkpoint_metadata

DEFAULT_MAIN_CACHE = Path("/pfs/mayuema/dense/wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40")
DEFAULT_NORMAL_CACHE = Path("/pfs/mayuema/dense/wan/cache/wan22_long477_normal_flip_yz_v1_bf16_bright40")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code_patch_path", required=True)
    parser.add_argument("--main_cache", type=Path, default=DEFAULT_MAIN_CACHE)
    parser.add_argument("--normal_cache", type=Path, default=DEFAULT_NORMAL_CACHE)
    parser.add_argument("--num_cases", type=int, default=20)
    parser.add_argument("--selection_seed", default="long477-gt-eval-v1")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_dir", default="/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B")
    parser.add_argument("--prompt_dir", default="/pfs/mayuema/dense/prompts")
    parser.add_argument("--decode_metrics", action="store_true")
    return parser.parse_args()


def load_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def selected_ids(root: Path, count: int, seed: str) -> list[str]:
    source_index = json.loads((root / "source_index.json").read_text(encoding="utf-8"))
    ids = [sample["id"] for sample in source_index["samples"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip IDs in source index")
    if count < 1 or count > len(ids):
        raise ValueError(f"num_cases must be within [1, {len(ids)}]")
    return sorted(ids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest())[:count]


def load_latent(path: Path, contract: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("cache_contract_hash") != contract:
        raise ValueError(f"Cache contract mismatch: {path}")
    latent = payload["latent"]
    if tuple(latent.shape) != long_infer.LONG_LATENT_SHAPE:
        raise ValueError(f"Unexpected latent shape {tuple(latent.shape)}: {path}")
    if latent.dtype != torch.bfloat16 or not torch.isfinite(latent).all():
        raise ValueError(f"Invalid latent: {path}")
    return latent


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    error = prediction.float() - target.float()
    per_frame_mse = error.square().mean(dim=(0, 2, 3))
    per_frame_mae = error.abs().mean(dim=(0, 2, 3))
    target_energy = target.float().square().mean().clamp_min(1e-12)
    pred_delta = prediction.float()[:, 1:] - prediction.float()[:, :-1]
    target_delta = target.float()[:, 1:] - target.float()[:, :-1]
    delta_error = pred_delta - target_delta
    x = torch.arange(per_frame_mse.numel(), dtype=torch.float32)
    slope = ((x - x.mean()) * (per_frame_mse - per_frame_mse.mean())).sum()
    slope = slope / (x - x.mean()).square().sum()
    return {
        "mse": float(per_frame_mse.mean()),
        "mae": float(per_frame_mae.mean()),
        "nmse_target_energy": float(per_frame_mse.mean() / target_energy),
        "mse_first_quarter": float(per_frame_mse[:30].mean()),
        "mse_middle_half": float(per_frame_mse[30:90].mean()),
        "mse_last_quarter": float(per_frame_mse[90:].mean()),
        "mse_late_over_early": float(per_frame_mse[90:].mean() / per_frame_mse[:30].mean().clamp_min(1e-12)),
        "mse_time_slope": float(slope),
        "temporal_delta_mse": float(delta_error.square().mean()),
        "temporal_delta_mae": float(delta_error.abs().mean()),
        "prediction_delta_l1": float(pred_delta.abs().mean()),
        "target_delta_l1": float(target_delta.abs().mean()),
        "per_frame_mse": [float(value) for value in per_frame_mse],
    }


def decoded_metrics(prediction: torch.Tensor, target: torch.Tensor, task: str) -> dict:
    totals = {"squared": 0.0, "absolute": 0.0, "count": 0, "first_squared": 0.0,
              "first_count": 0, "last_squared": 0.0, "last_count": 0,
              "delta_squared": 0.0, "delta_count": 0, "angle": 0.0,
              "angle_count": 0, "angle_lt_11_25": 0}
    previous_prediction = previous_target = None
    for start in range(0, prediction.shape[1], 8):
        end = min(start + 8, prediction.shape[1])
        pred = prediction[:, start:end].clamp(-1, 1)
        gt = target[:, start:end].clamp(-1, 1)
        if task == "depth":
            pred = pred.mean(dim=0, keepdim=True)
            gt = gt.mean(dim=0, keepdim=True)
        error = pred - gt
        totals["squared"] += float(error.square().sum())
        totals["absolute"] += float(error.abs().sum())
        totals["count"] += error.numel()
        if start < 120:
            stop = min(end, 120) - start
            totals["first_squared"] += float(error[:, :stop].square().sum())
            totals["first_count"] += error[:, :stop].numel()
        if end > 357:
            begin = max(357, start) - start
            totals["last_squared"] += float(error[:, begin:].square().sum())
            totals["last_count"] += error[:, begin:].numel()
        pred_with_previous = pred if previous_prediction is None else torch.cat((previous_prediction, pred), dim=1)
        gt_with_previous = gt if previous_target is None else torch.cat((previous_target, gt), dim=1)
        if pred_with_previous.shape[1] > 1:
            delta_error = (pred_with_previous[:, 1:] - pred_with_previous[:, :-1]) - (gt_with_previous[:, 1:] - gt_with_previous[:, :-1])
            totals["delta_squared"] += float(delta_error.square().sum())
            totals["delta_count"] += delta_error.numel()
        previous_prediction = pred[:, -1:].clone()
        previous_target = gt[:, -1:].clone()
        if task == "normal":
            pred_unit = torch.nn.functional.normalize(pred, dim=0, eps=1e-6)
            gt_unit = torch.nn.functional.normalize(gt, dim=0, eps=1e-6)
            angle = torch.rad2deg(torch.acos((pred_unit * gt_unit).sum(dim=0).clamp(-1, 1)))
            totals["angle"] += float(angle.sum())
            totals["angle_count"] += angle.numel()
            totals["angle_lt_11_25"] += int((angle < 11.25).sum())
    mse = totals["squared"] / totals["count"]
    result = {
        "pixel_mse": mse,
        "pixel_mae": totals["absolute"] / totals["count"],
        "pixel_psnr": 10.0 * float(torch.log10(torch.tensor(4.0 / max(mse, 1e-12)))),
        "pixel_mse_first_quarter": totals["first_squared"] / totals["first_count"],
        "pixel_mse_last_quarter": totals["last_squared"] / totals["last_count"],
        "pixel_temporal_delta_mse": totals["delta_squared"] / totals["delta_count"],
    }
    if task == "normal":
        result["normal_angular_mean_deg"] = totals["angle"] / totals["angle_count"]
        result["normal_within_11_25_percent"] = 100.0 * totals["angle_lt_11_25"] / totals["angle_count"]
    return result


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    main_manifest = load_manifest(args.main_cache)
    normal_manifest = load_manifest(args.normal_cache)
    for key in ("num_clips", "ordered_clip_ids_hash", "source_index_sha256"):
        if main_manifest.get(key) != normal_manifest.get(key):
            raise ValueError(f"Main and canonical-normal caches differ: {key}")
    ids = selected_ids(args.main_cache, args.num_cases, args.selection_seed)
    shard_ids = ids[args.shard_index :: args.num_shards]
    if not shard_ids:
        raise ValueError("Empty shard")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    checkpoint = Path(args.checkpoint).resolve()
    metadata = checkpoint_metadata(checkpoint)
    training_mode = long_infer.validate_ar_checkpoint_metadata(metadata)
    ar_train.dense4.LATENT_SHAPE = long_infer.LONG_LATENT_SHAPE
    ar_train.AR_CONFIG = long_infer.LONG_AR_CONFIG
    prompt_texts = json.loads(metadata["output_prompt_texts"])
    prompt_ids = {task: common.prompt_id_for_task(prompt_texts, task, "person") for task in ("depth", "normal")}
    module = ar_train.ARTeacherForcingDense4LoRAModule(
        args.model_dir, int(metadata["lora_rank"]), metadata["lora_targets"],
        int(metadata["router_hidden_dim"]), metadata["router_mode"], 0.0,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(prompt_texts, device)
    args.official_repo = "/pfs/mayuema/dense/wan/Wan2.2"
    args.depth_cache_dir = "/pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40"
    args.target_cache_dir = "/pfs/mayuema/dense/wan/cache/wan22_dense4_targets_bf16_bright40"
    provenance = long_infer.checkpoint_provenance(module, metadata, args)
    module.pipe.dit.to(device).eval()
    loaded = ar_train._load_dense4_checkpoint(
        module=module, path=checkpoint, expected_provenance=provenance,
        lora_rank=int(metadata["lora_rank"]), lora_targets=metadata["lora_targets"],
        router_hidden_dim=int(metadata["router_hidden_dim"]), expected_router_mode=metadata["router_mode"],
    )
    vae = (
        common.load_official_vae(Path(args.official_repo), Path(args.model_dir), device)
        if args.decode_metrics else None
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        header = {
            "schema": "wan22_long477_paired_gt_eval_v1", "record_type": "header",
            "evaluation_scope": "in_domain_training_pool", "label": args.label,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_step": int(loaded["step"]), "training_mode": training_mode,
            "selection_seed": args.selection_seed, "selected_clip_ids": ids,
            "num_cases": args.num_cases, "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "main_cache_contract_hash": main_manifest["cache_contract_hash"],
            "normal_cache_contract_hash": normal_manifest["cache_contract_hash"],
        }
        handle.write(json.dumps(header) + "\n")
        handle.flush()
        print(json.dumps(header), flush=True)
        for clip_id in shard_ids:
            source = load_latent(args.main_cache / "source" / f"{clip_id}.pt", main_manifest["cache_contract_hash"])
            for task in ("depth", "normal"):
                target_root = args.normal_cache if task == "normal" else args.main_cache
                target_manifest = normal_manifest if task == "normal" else main_manifest
                target = load_latent(target_root / task / f"{clip_id}.pt", target_manifest["cache_contract_hash"])
                started = time.perf_counter()
                try:
                    prediction, gates = module.predict_free_running(
                        source.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                        torch.tensor([common.TASK_TO_ID[task]], device=device),
                        torch.tensor([prompt_ids[task]], device=device),
                    )
                    prediction = prediction[0].cpu()
                finally:
                    module.task_router.clear()
                row = {
                    "record_type": "sample", "label": args.label, "clip_id": clip_id,
                    "task": task, "elapsed_seconds": time.perf_counter() - started,
                    "gate_mean": float(gates.mean()), **metrics(prediction, target),
                }
                if vae is not None:
                    prediction_decoded = long_infer.decode_long_video(vae, prediction, device)
                    target_decoded = long_infer.decode_long_video(vae, target, device)
                    row.update(decoded_metrics(prediction_decoded, target_decoded, task))
                    del prediction_decoded, target_decoded
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(json.dumps({k: row[k] for k in ("label", "clip_id", "task", "mse", "mse_last_quarter", "elapsed_seconds")}), flush=True)
                del prediction, gates, target
                torch.cuda.empty_cache()
            del source


if __name__ == "__main__":
    main()
