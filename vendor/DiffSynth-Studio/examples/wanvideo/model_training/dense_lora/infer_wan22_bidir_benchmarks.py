"""Evaluate the original bidirectional Wan2.2 Dense4 checkpoint on image sets.

Each independent benchmark image is center-cropped to 26:15, resized to
832x480 without distortion, and repeated for all 81 input frames.  Only the
middle decoded frame is evaluated.  This is our explicitly versioned protocol,
not GenCeption's direct-resize protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from infer_wan22_dense4 import (  # noqa: E402
    Dense4LoRAModule,
    build_provenance_metadata,
    checkpoint_metadata,
    decode_video,
    encode_video,
    load_dense4_checkpoint,
    load_official_vae,
    prompt_id_for_task,
)
from wan22_bidir_benchmark_protocol import (  # noqa: E402
    CENTER_FRAME,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    align_relative_inverse_depth,
    center_crop_box,
    crop_array,
    depth_metrics,
    image_to_repeated_video,
    inverse_depth_to_depth,
    matte_metrics,
    normal_metrics,
    relative_inverse_depth_metrics,
    resize_prediction_to_crop,
)
from wan22_task_router import TASK_TO_ID  # noqa: E402


BENCHMARK_ROOT = Path("/pfs/mayuema/dense/benchmarks")
SPLIT_ROOT = Path("/pfs/mayuema/dense/baselines/Edit2Perceive/data_split")
WAN_ROOT = Path("/pfs/mayuema/dense/wan")


@dataclass(frozen=True)
class Sample:
    index: int
    name: str
    image_path: Path
    ground_truth_path: Path
    task: str


DATASETS = {
    "kitti": (
        "depth",
        BENCHMARK_ROOT / "depth/kitti",
        SPLIT_ROOT / "kitti_depth/eigen_test_files_with_gt_cleaned.txt",
    ),
    "eth3d": (
        "depth",
        BENCHMARK_ROOT / "depth/eth3d",
        SPLIT_ROOT / "eth3d_depth/eth3d_filename_list.txt",
    ),
    "nyuv2": (
        "normal",
        BENCHMARK_ROOT / "normals/nyuv2/test",
        SPLIT_ROOT / "nyu_normals/nyuv2_test.txt",
    ),
    "scannet": (
        "normal",
        BENCHMARK_ROOT / "normals/scannet",
        SPLIT_ROOT / "scannet_normals/scannet_test.txt",
    ),
    "ibims": (
        "normal",
        BENCHMARK_ROOT / "normals/ibims/ibims",
        SPLIT_ROOT / "ibims_normals/ibims_test.txt",
    ),
    "diode": (
        "normal",
        BENCHMARK_ROOT / "normals/diode",
        SPLIT_ROOT / "diode_normals/diode_test.txt",
    ),
    "p3m500np": (
        "matte",
        BENCHMARK_ROOT / "matting/P3M-10k/P3M-10k",
        SPLIT_ROOT / "P3M_matting/filenames_val_NP.txt",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument(
        "--checkpoint",
        default=str(
            WAN_ROOT
            / "outputs/wan22_dense4_hyperlora_lingbot_stage2_20k/step-20000.safetensors"
        ),
    )
    parser.add_argument("--model_dir", default=str(WAN_ROOT / "Wan2.2-TI2V-5B"))
    parser.add_argument("--official_repo", default=str(WAN_ROOT / "Wan2.2"))
    parser.add_argument("--prompt_dir", default="/pfs/mayuema/dense/prompts")
    parser.add_argument(
        "--depth_cache_dir", default=str(WAN_ROOT / "cache/wan22_depth_latents_bf16_bright40")
    )
    parser.add_argument(
        "--target_cache_dir",
        default=str(WAN_ROOT / "cache/wan22_dense4_targets_bf16_bright40"),
    )
    parser.add_argument(
        "--code_patch_path",
        default=str(WAN_ROOT / "docs/wan22_dense4_lingbot_stage2_runtime_source.patch"),
    )
    parser.add_argument(
        "--normal_transform",
        choices=("identity", "flip_x", "flip_yz"),
        default="identity",
    )
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--recompute_existing_depth_metrics",
        action="store_true",
        help="Recompute depth metrics from saved raw predictions without loading the model.",
    )
    return parser.parse_args()


def load_samples(dataset: str) -> tuple[str, list[Sample]]:
    task, root, split = DATASETS[dataset]
    rows = [line.split() for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    for index, row in enumerate(rows):
        image_rel = row[0]
        gt_rel = row[-1] if task == "matte" else row[1]
        image_path = root / image_rel
        gt_path = root / gt_rel
        if not image_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(f"Missing benchmark pair: {image_path}, {gt_path}")
        samples.append(
            Sample(
                index=index,
                name=f"{index:06d}_{Path(image_rel).stem}",
                image_path=image_path,
                ground_truth_path=gt_path,
                task=task,
            )
        )
    return task, samples


def load_ground_truth(sample: Sample, image_size: tuple[int, int]) -> np.ndarray:
    if sample.task == "normal":
        return np.load(sample.ground_truth_path)
    if sample.task == "matte":
        return np.asarray(Image.open(sample.ground_truth_path).convert("L"), dtype=np.float32) / 255.0
    if sample.ground_truth_path.suffix.lower() == ".jpg":
        width, height = image_size
        value = np.fromfile(sample.ground_truth_path, dtype=np.float32)
        if value.size != width * height:
            raise ValueError(
                f"ETH3D depth size {value.size} != {width}*{height}: {sample.ground_truth_path}"
            )
        return value.reshape(height, width)
    return np.asarray(Image.open(sample.ground_truth_path), dtype=np.float32) / 256.0


def transform_normal_gt(value: np.ndarray, transform: str) -> np.ndarray:
    value = value.astype(np.float32, copy=True)
    if transform == "flip_x":
        value[..., 0] *= -1
    elif transform == "flip_yz":
        value[..., 1:] *= -1
    return value


def initialize_model(args: argparse.Namespace, task: str, device: torch.device):
    checkpoint = Path(args.checkpoint).resolve()
    metadata = checkpoint_metadata(checkpoint)
    prompt_texts = json.loads(metadata["output_prompt_texts"])
    prompt_id = prompt_id_for_task(prompt_texts, task, "person")
    module = Dense4LoRAModule(
        args.model_dir,
        int(metadata["lora_rank"]),
        metadata["lora_targets"],
        int(metadata["router_hidden_dim"]),
        metadata["router_mode"],
        0.0,
    )
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(prompt_texts, device)
    provenance = build_provenance_metadata(
        module,
        model_dir=Path(args.model_dir).resolve(),
        prompt_texts=prompt_texts,
        prompt_dir=args.prompt_dir,
        depth_cache_dir=args.depth_cache_dir,
        target_cache_dir=args.target_cache_dir,
        code_patch_path=args.code_patch_path,
        repo_dir=THIS_DIR.parents[3],
    )
    # The new benchmark driver is Git-recorded, while the audited model source
    # remains identified by the checkpoint's recorded revision and patch hash.
    provenance["diffsynth_git_head"] = metadata["diffsynth_git_head"]
    module.provenance_metadata = provenance
    module.pipe.dit.to(device).eval()
    loaded = load_dense4_checkpoint(
        module,
        checkpoint,
        expected_provenance=provenance,
        lora_rank=int(metadata["lora_rank"]),
        lora_targets=metadata["lora_targets"],
        router_hidden_dim=int(metadata["router_hidden_dim"]),
        expected_router_mode=metadata["router_mode"],
    )
    vae = load_official_vae(Path(args.official_repo).resolve(), Path(args.model_dir), device)
    return module, vae, prompt_id, loaded


def evaluate_prediction(
    args: argparse.Namespace,
    sample: Sample,
    decoded: torch.Tensor,
    ground_truth: np.ndarray,
    box,
) -> tuple[np.ndarray, dict[str, float]]:
    center = decoded[:, CENTER_FRAME].permute(1, 2, 0).numpy()
    gt_crop = crop_array(ground_truth, box)
    if sample.task == "depth":
        prediction = resize_prediction_to_crop(center.mean(axis=-1), box)
        return prediction, evaluate_depth_prediction(prediction, gt_crop, args.max_depth)
    if sample.task == "normal":
        prediction = resize_prediction_to_crop(center, box)
        gt_crop = transform_normal_gt(gt_crop, args.normal_transform)
        return prediction, normal_metrics(prediction, gt_crop)
    prediction = resize_prediction_to_crop((center.mean(axis=-1) + 1.0) / 2.0, box)
    return prediction, matte_metrics(prediction, gt_crop)


def evaluate_depth_prediction(
    prediction: np.ndarray, ground_truth_depth: np.ndarray, max_depth: float
) -> dict[str, float]:
    valid = (
        np.isfinite(ground_truth_depth)
        & (ground_truth_depth > 1e-3)
        & (ground_truth_depth < max_depth)
    )
    aligned_inverse, scale, shift = align_relative_inverse_depth(
        prediction, ground_truth_depth, valid
    )
    metrics = relative_inverse_depth_metrics(aligned_inverse, ground_truth_depth, valid)
    metrics.update(
        depth_metrics(
            inverse_depth_to_depth(aligned_inverse, max_depth=max_depth),
            ground_truth_depth,
            valid,
        )
    )
    metrics.update({"inverse_scale": scale, "inverse_shift": shift})
    return metrics


def recompute_existing_depth_metrics(
    args: argparse.Namespace, samples: list[Sample], output_root: Path
) -> None:
    rows = []
    for sample in samples:
        sample_dir = output_root / sample.name
        prediction_path = sample_dir / "prediction.npy"
        metrics_path = sample_dir / "metrics.json"
        if not prediction_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Missing existing prediction or metadata: {sample_dir}")
        image = Image.open(sample.image_path)
        box = center_crop_box(*image.size)
        ground_truth = load_ground_truth(sample, image.size)
        gt_crop = crop_array(ground_truth, box)
        prediction = np.load(prediction_path).astype(np.float32)
        if prediction.shape != gt_crop.shape:
            raise ValueError(
                f"Saved prediction/GT shape mismatch {prediction.shape} vs {gt_crop.shape}: {sample.name}"
            )
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        row["metrics"] = evaluate_depth_prediction(prediction, gt_crop, args.max_depth)
        row["depth_evaluation_revision"] = "relative_inverse_depth_affine_v2"
        metrics_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)

    aggregate = {
        key: float(np.mean([row["metrics"][key] for row in rows]))
        for key in rows[0]["metrics"]
    }
    shard_path = output_root / f"shard_{args.shard_index:02d}.json"
    manifest = json.loads(shard_path.read_text(encoding="utf-8"))
    manifest["schema"] = "wan22_bidir_image_benchmark_center_crop_v2"
    manifest["protocol"]["depth_alignment"] = (
        "per_image_affine_alignment_to_gt_inverse_depth"
    )
    manifest["protocol"]["depth_primary_representation"] = "relative_inverse_depth"
    manifest["protocol"]["depth_domain_metrics"] = (
        "reciprocal_of_aligned_inverse_depth_for_standard_comparison_only"
    )
    manifest["aggregate"] = aggregate
    shard_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    task, all_samples = load_samples(args.dataset)
    if args.max_samples is not None:
        all_samples = all_samples[: args.max_samples]
    samples = all_samples[args.shard_index :: args.num_shards]
    if not samples:
        raise ValueError("Shard has no samples")

    output_root = Path(args.output_root).resolve() / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    if args.recompute_existing_depth_metrics:
        if task != "depth":
            raise ValueError("Existing metric recomputation is only valid for depth datasets")
        recompute_existing_depth_metrics(args, samples, output_root)
        return
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    module, vae, prompt_id, loaded = initialize_model(args, task, device)
    task_id = TASK_TO_ID[task]
    print(json.dumps({"ready": True, "dataset": args.dataset, "samples": len(samples)}), flush=True)

    rows = []
    for sample in samples:
        sample_dir = output_root / sample.name
        complete = sample_dir / "COMPLETE"
        if complete.is_file():
            rows.append(json.loads((sample_dir / "metrics.json").read_text()))
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(sample.image_path).convert("RGB")
        video, box = image_to_repeated_video(image)
        ground_truth = load_ground_truth(sample, image.size)
        if ground_truth.shape[:2] != (image.height, image.width):
            raise ValueError(
                f"RGB/GT size mismatch {image.size} vs {ground_truth.shape}: {sample.name}"
            )
        source_latent = encode_video(vae, video, device)
        with torch.inference_mode():
            try:
                velocity, gates = module.predict_velocity(
                    source_latent.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
                    torch.tensor([task_id], device=device),
                    torch.tensor([prompt_id], device=device),
                )
            finally:
                module.task_router.clear()
            prediction_latent = (-velocity[0]).detach().to(device="cpu", dtype=torch.bfloat16)
            decoded = decode_video(vae, prediction_latent, device)
        prediction, metrics = evaluate_prediction(args, sample, decoded, ground_truth, box)
        row = {
            "dataset": args.dataset,
            "index": sample.index,
            "name": sample.name,
            "image": str(sample.image_path),
            "ground_truth": str(sample.ground_truth_path),
            "crop_box_ltrb": [box.left, box.top, box.right, box.bottom],
            "metrics": metrics,
            "gate_mean": float(gates.mean()),
        }
        np.save(sample_dir / "prediction.npy", prediction.astype(np.float16))
        (sample_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        complete.write_text(json.dumps({"checkpoint_step": int(loaded["step"])}), encoding="utf-8")
        rows.append(row)
        print(json.dumps({"completed": sample.name, **metrics}), flush=True)
        del source_latent, velocity, gates, prediction_latent, decoded
        torch.cuda.empty_cache()

    aggregate = {
        key: float(np.mean([row["metrics"][key] for row in rows]))
        for key in rows[0]["metrics"]
    }
    manifest = {
        "schema": "wan22_bidir_image_benchmark_center_crop_v2",
        "dataset": args.dataset,
        "task": task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(loaded["step"]),
        "protocol": {
            "spatial": "largest_center_crop_26_15_then_bicubic_resize_832x480",
            "temporal": "repeat_each_independent_image_81_frames_take_middle_frame_40",
            "prediction_restore": "bilinear_to_uncropped_gt_crop_resolution",
            "depth_alignment": "per_image_affine_alignment_to_gt_inverse_depth",
            "depth_primary_representation": "relative_inverse_depth",
            "depth_domain_metrics": "reciprocal_of_aligned_inverse_depth_for_standard_comparison_only",
            "normal_gt_transform": args.normal_transform,
        },
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_samples": len(rows),
        "aggregate": aggregate,
    }
    (output_root / f"shard_{args.shard_index:02d}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
