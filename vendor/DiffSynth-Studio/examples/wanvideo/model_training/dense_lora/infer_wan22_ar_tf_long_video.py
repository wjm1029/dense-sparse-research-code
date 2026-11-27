"""Free-running chunkwise AR inference for exact 477-frame Wan2.2 clips.

This is a separate inference entry point: official Wan/DiffSynth sources and
the audited 81-frame training implementation are not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import infer_wan22_dense4 as common
import train_wan22_ar_tf_ddp as ar_train
from wan22_ar_teacher_forcing import ARTeacherForcingConfig
from wan22_cache_integrity import sha256_file
from wan22_cache_intervention import (
    StaticCacheInterventionConfig,
    build_static_cache_intervention_schedule,
)
from wan22_dense4_checkpoint import (
    OPTIONAL_PROVENANCE_KEYS,
    REQUIRED_PROVENANCE_KEYS,
    build_provenance_metadata,
    checkpoint_metadata,
)


RAW_FRAMES = 477
LATENT_FRAMES = 120
LONG_LATENT_SHAPE = (48, LATENT_FRAMES, 30, 52)
LONG_AR_CONFIG = ARTeacherForcingConfig(
    num_frames=LATENT_FRAMES, frames_per_chunk=3, latent_channels=48
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--matte_subject", default="person")
    parser.add_argument("--model_dir", default="/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B")
    parser.add_argument("--official_repo", default="/pfs/mayuema/dense/wan/Wan2.2")
    parser.add_argument("--prompt_dir", default="/pfs/mayuema/dense/prompts")
    parser.add_argument(
        "--depth_cache_dir",
        default="/pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40",
    )
    parser.add_argument(
        "--target_cache_dir",
        default="/pfs/mayuema/dense/wan/cache/wan22_dense4_targets_bf16_bright40",
    )
    parser.add_argument(
        "--code_patch_path",
        default="/pfs/mayuema/dense/wan/docs/wan22_ar_tf_chunk3_runtime_source.patch",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache_intervention", action="store_true")
    parser.add_argument("--cache_static_enter_threshold", type=float, default=0.5)
    parser.add_argument("--cache_static_exit_threshold", type=float, default=1.0)
    parser.add_argument("--cache_static_confirm_chunks", type=int, default=3)
    parser.add_argument("--cache_static_max_hold_chunks", type=int, default=20)
    parser.add_argument("--cache_scene_cut_threshold", type=float, default=20.0)
    return parser.parse_args()


def read_long_video(path: Path) -> tuple[np.ndarray, float]:
    reader = imageio.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        frames = np.stack([np.asarray(frame)[..., :3] for frame in reader]).astype(np.uint8)
    finally:
        reader.close()
    expected = (RAW_FRAMES, 480, 832, 3)
    if tuple(frames.shape) != expected:
        raise ValueError(f"Expected {expected}, got {tuple(frames.shape)}: {path}")
    fps = float(metadata.get("fps") or 16.0)
    if abs(fps - 16.0) > 1e-6:
        raise ValueError(f"Expected 16 fps, got {fps}: {path}")
    return frames, fps


@torch.no_grad()
def encode_long_video(vae, frames: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(frames.copy()).float().div_(127.5).sub_(1.0)
    value = value.permute(3, 0, 1, 2).contiguous().to(device=device, dtype=torch.bfloat16)
    encoded = vae.encode([value])
    if not encoded or encoded[0] is None:
        raise RuntimeError("Official Wan2.2 VAE encode returned no latent")
    latent = encoded[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(latent.shape) != LONG_LATENT_SHAPE or not torch.isfinite(latent).all():
        raise ValueError(f"Invalid long-video latent: {tuple(latent.shape)}")
    return latent


@torch.no_grad()
def decode_long_video(vae, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    decoded = vae.decode([latent.to(device=device, dtype=torch.bfloat16)])
    if not decoded or decoded[0] is None:
        raise RuntimeError("Official Wan2.2 VAE decode returned no video")
    value = decoded[0].detach().float().cpu().contiguous()
    expected = (3, RAW_FRAMES, 480, 832)
    if tuple(value.shape) != expected or not torch.isfinite(value).all():
        raise ValueError(f"Invalid decoded long video: {tuple(value.shape)}")
    return value


def write_outputs(
    output_dir: Path,
    task: str,
    source: np.ndarray,
    prediction: torch.Tensor,
    latent: torch.Tensor,
    fps: float,
) -> None:
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    predicted = common.task_frames(prediction, task)
    comparisons = np.stack(
        [
            np.concatenate(
                (common.label(source[i], "RGB input"), common.label(predicted[i], f"AR: {task}")),
                axis=1,
            )
            for i in range(RAW_FRAMES)
        ]
    )
    common.write_video(task_dir / "prediction.mp4", predicted, fps)
    common.write_video(task_dir / "comparison.mp4", comparisons, fps)
    indices = np.linspace(0, RAW_FRAMES - 1, 6, dtype=int)
    Image.fromarray(np.concatenate([comparisons[i] for i in indices], axis=0)).save(
        task_dir / "preview.png"
    )
    torch.save(
        {"schema": "wan22_dense4_long_ar_latent_v1", "task": task, "latent": latent},
        task_dir / "prediction_latent.pt",
    )


def checkpoint_provenance(module, metadata: dict[str, str], args) -> dict[str, str]:
    patch = Path(args.code_patch_path).resolve()
    if sha256_file(patch) != metadata.get("runtime_source_patch_sha256"):
        raise ValueError("Inference source patch does not match checkpoint metadata")
    if metadata.get("self_forcing") == "previous_predicted_chunk_stopgrad_v1":
        keys = REQUIRED_PROVENANCE_KEYS | OPTIONAL_PROVENANCE_KEYS
        return {key: metadata[key] for key in keys if key in metadata}
    result = build_provenance_metadata(
        module,
        model_dir=Path(args.model_dir).resolve(),
        prompt_texts=json.loads(metadata["output_prompt_texts"]),
        prompt_dir=args.prompt_dir,
        depth_cache_dir=args.depth_cache_dir,
        target_cache_dir=args.target_cache_dir,
        code_patch_path=patch,
        repo_dir=THIS_DIR.parents[3],
    )
    result["diffsynth_git_head"] = metadata["diffsynth_git_head"]
    return result


def validate_ar_checkpoint_metadata(metadata: dict[str, str]) -> str:
    if metadata.get("ar_frames_per_chunk") != "3":
        raise ValueError("Checkpoint was not trained with 3-frame latent chunks")
    if metadata.get("self_forcing") == "previous_predicted_chunk_stopgrad_v1":
        expected = {
            "ar_teacher_forcing": "none",
            "ar_num_frames": str(LATENT_FRAMES),
            "ar_num_chunks": str(LATENT_FRAMES // LONG_AR_CONFIG.frames_per_chunk),
            "causal_num_frames": str(LATENT_FRAMES),
            "causal_frames_per_chunk": str(LONG_AR_CONFIG.frames_per_chunk),
            "bptt": "chunkwise_truncated_stopgrad_kv_and_history_v1",
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Invalid long self-forcing checkpoint metadata: {mismatches}")
        attention_config = (
            metadata.get("causal_local_attn_frames", "-1"),
            metadata.get("causal_sink_frames", "0"),
        )
        if attention_config not in (("-1", "0"), ("12", "3")):
            raise ValueError(
                "Unsupported causal attention checkpoint metadata: "
                f"local/sink={attention_config}"
            )
        return "self_forcing"
    if metadata.get("ar_teacher_forcing") == "previous_gt_chunk_v1":
        return "teacher_forcing"
    raise ValueError("Checkpoint is neither audited AR teacher-forcing nor self-forcing")


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if not cases:
        raise ValueError("Selection manifest contains no cases")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    cases = cases[args.shard_index :: args.num_shards]
    if not cases:
        raise ValueError("This shard contains no cases")
    tasks = common.selected_tasks(args.tasks)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Long AR inference requires CUDA")
    torch.cuda.set_device(device)

    checkpoint = Path(args.checkpoint).resolve()
    metadata = checkpoint_metadata(checkpoint)
    training_mode = validate_ar_checkpoint_metadata(metadata)

    # The weights are length-agnostic. Install the same causal/AR operators with
    # a longer audited sequence contract before constructing the wrapper class.
    ar_train.dense4.LATENT_SHAPE = LONG_LATENT_SHAPE
    ar_train.AR_CONFIG = LONG_AR_CONFIG
    model_dir = Path(args.model_dir).resolve()
    prompt_texts = json.loads(metadata["output_prompt_texts"])
    prompt_ids = {
        task: common.prompt_id_for_task(prompt_texts, task, args.matte_subject) for task in tasks
    }
    if metadata.get("causal_local_attn_frames", "-1") == "12":
        from wan22_opsd_local12_inference import OPSDLocal12InferenceModule

        module_class = OPSDLocal12InferenceModule
        attention_description = "local12_sink3_prerope_fixed_pool_per_layer_kv_cache"
    else:
        module_class = ar_train.ARTeacherForcingDense4LoRAModule
        attention_description = "full_prefix_per_layer_kv_cache"
    if args.cache_intervention and metadata.get("causal_local_attn_frames", "-1") != "12":
        raise ValueError("Cache intervention requires a local-12 bounded-attention checkpoint")
    module = module_class(
        str(model_dir),
        int(metadata["lora_rank"]),
        metadata["lora_targets"],
        int(metadata["router_hidden_dim"]),
        metadata["router_mode"],
        0.0,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(prompt_texts, device)
    provenance = checkpoint_provenance(module, metadata, args)
    module.pipe.dit.to(device).eval()
    loaded = ar_train._load_dense4_checkpoint(
        module=module,
        path=checkpoint,
        expected_provenance=provenance,
        lora_rank=int(metadata["lora_rank"]),
        lora_targets=metadata["lora_targets"],
        router_hidden_dim=int(metadata["router_hidden_dim"]),
        expected_router_mode=metadata["router_mode"],
    )
    vae = common.load_official_vae(Path(args.official_repo).resolve(), model_dir, device)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "resident_long_ar_ready": True,
                "training_mode": training_mode,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "cases": len(cases),
            }
        ),
        flush=True,
    )

    for case in cases:
        source_path = Path(case["path"]).resolve()
        output_dir = output_root / case["case_name"]
        if (output_dir / "INFERENCE_COMPLETE").is_file() and not args.overwrite:
            print(json.dumps({"skip_complete": case["case_name"]}), flush=True)
            continue
        common.prepare_output_dir(output_dir, overwrite=args.overwrite)
        input_frames, fps = read_long_video(source_path)
        source_latent = encode_long_video(vae, input_frames, device)
        intervention_schedule = None
        intervention_report = {"enabled": False}
        if args.cache_intervention:
            intervention_config = StaticCacheInterventionConfig(
                enter_threshold=args.cache_static_enter_threshold,
                exit_threshold=args.cache_static_exit_threshold,
                confirm_chunks=args.cache_static_confirm_chunks,
                max_hold_chunks=args.cache_static_max_hold_chunks,
                scene_cut_threshold=args.cache_scene_cut_threshold,
            )
            intervention_schedule, intervention_report = (
                build_static_cache_intervention_schedule(
                    input_frames, LONG_AR_CONFIG.num_chunks, intervention_config
                )
            )
            intervention_report["enabled"] = True
            (output_dir / "cache_intervention.json").write_text(
                json.dumps(intervention_report, indent=2), encoding="utf-8"
            )
        rows = []
        for task in tasks:
            started = time.perf_counter()
            task_id = common.TASK_TO_ID[task]
            if hasattr(module.causal_controller, "set_cache_intervention_schedule"):
                module.causal_controller.set_cache_intervention_schedule(intervention_schedule)
            try:
                prediction, gates = module.predict_free_running(
                    source_latent.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                    torch.tensor([task_id], device=device, dtype=torch.long),
                    torch.tensor([prompt_ids[task]], device=device, dtype=torch.long),
                )
                prediction_latent = prediction[0].detach().to(
                    device="cpu", dtype=torch.bfloat16
                ).contiguous()
            finally:
                module.task_router.clear()
            if tuple(prediction_latent.shape) != LONG_LATENT_SHAPE:
                raise ValueError(f"Invalid {task} latent shape: {tuple(prediction_latent.shape)}")
            decoded = decode_long_video(vae, prediction_latent, device)
            write_outputs(output_dir, task, input_frames, decoded, prediction_latent, fps)
            elapsed = time.perf_counter() - started
            row = {
                "task": task,
                "elapsed_seconds": elapsed,
                "output_fps": RAW_FRAMES / elapsed,
                "gate_min": float(gates.min()),
                "gate_max": float(gates.max()),
                "gate_mean": float(gates.mean()),
            }
            if hasattr(module.causal_controller, "cache_intervention_stats"):
                row["cache_intervention"] = module.causal_controller.cache_intervention_stats()
            rows.append(row)
            print(json.dumps({"case": case["case_name"], **row}), flush=True)
            del prediction, prediction_latent, decoded, gates
            torch.cuda.empty_cache()
        result = {
            "schema": "wan22_dense4_long_free_running_ar_inference_v1",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_step": int(loaded["step"]),
            "source": str(source_path),
            "source_frames": RAW_FRAMES,
            "source_latent_shape": list(source_latent.shape),
            "fps": fps,
            "ar": {
                "mode": "free_running_previous_predicted_chunk",
                "latent_frames_per_chunk": 3,
                "raw_frames_per_chunk": 12,
                "latent_chunks": LONG_AR_CONFIG.num_chunks,
                "causal_attention": attention_description,
                "cache_intervention": intervention_report,
            },
            "tasks": rows,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "INFERENCE_COMPLETE").write_text(
            json.dumps({"checkpoint_step": int(loaded["step"]), "tasks": tasks}),
            encoding="utf-8",
        )
        print(json.dumps({"completed": case["case_name"], "tasks": tasks}), flush=True)
        del source_latent, input_frames
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
