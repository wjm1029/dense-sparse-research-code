#!/usr/bin/env python3
"""Measure matched bidirectional, causal, and free-running AR Ditto20 outputs.

Ditto20 has no ground-truth task labels.  Bidirectional output is therefore a
reference, not an accuracy target.  The report separates reference deviation
from intrinsic temporal stability and exposes per-latent-frame drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch


TASKS = ("depth", "normal", "matte", "raymap")
MODES = ("bid", "causal", "ar")
FRAMES_PER_CHUNK = 3


def load_latent(path: Path, task: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "wan22_dense4_inference_latent_v1":
        raise ValueError(f"Unexpected latent schema: {path}")
    if payload.get("task") != task:
        raise ValueError(f"Task mismatch: {path}")
    latent = payload.get("latent")
    if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != (48, 21, 30, 52):
        raise ValueError(f"Unexpected latent shape: {path}")
    if latent.dtype != torch.bfloat16 or not torch.isfinite(latent).all():
        raise ValueError(f"Invalid latent payload: {path}")
    return latent.float()


def load_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame.astype(np.float32) / 255.0)
    capture.release()
    if len(frames) != 81:
        raise ValueError(f"Expected 81 frames, got {len(frames)}: {path}")
    return np.stack(frames)


def mean_square(value: torch.Tensor) -> float:
    return float(value.square().mean())


def latent_metrics(value: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    error = value - reference
    frame_mse = error.square().mean(dim=(0, 2, 3))
    reference_variance = float((reference - reference.mean()).square().mean())
    deltas = value[:, 1:] - value[:, :-1]
    acceleration = value[:, 2:] - 2 * value[:, 1:-1] + value[:, :-2]
    transition_mse = deltas.square().mean(dim=(0, 2, 3))
    boundary_indices = torch.tensor(
        [index - 1 for index in range(FRAMES_PER_CHUNK, value.shape[1], FRAMES_PER_CHUNK)]
    )
    within_indices = torch.tensor(
        [index for index in range(value.shape[1] - 1) if index not in boundary_indices.tolist()]
    )
    boundary = float(transition_mse[boundary_indices].mean())
    within = float(transition_mse[within_indices].mean())
    early = float(frame_mse[:FRAMES_PER_CHUNK].mean())
    late = float(frame_mse[-FRAMES_PER_CHUNK:].mean())
    x = torch.arange(frame_mse.numel(), dtype=torch.float32)
    slope = float(((x - x.mean()) * (frame_mse - frame_mse.mean())).sum() / (x - x.mean()).square().sum())
    mse = mean_square(error)
    return {
        "mse_to_bid": mse,
        "nrmse_to_bid": math.sqrt(mse / max(reference_variance, 1e-12)),
        "early_chunk_mse_to_bid": early,
        "late_chunk_mse_to_bid": late,
        "late_over_early_mse": late / max(early, 1e-12),
        "mse_to_bid_slope_per_latent_frame": slope,
        "temporal_delta_mse": float(transition_mse.mean()),
        "temporal_acceleration_mse": mean_square(acceleration),
        "chunk_boundary_delta_mse": boundary,
        "within_chunk_delta_mse": within,
        "chunk_boundary_ratio": boundary / max(within, 1e-12),
        "per_latent_frame_mse_to_bid": [float(item) for item in frame_mse],
    }


def video_metrics(value: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = value - reference
    delta = np.diff(value, axis=0)
    acceleration = np.diff(value, n=2, axis=0)
    return {
        "decoded_mse_to_bid": float(np.mean(error * error)),
        "decoded_temporal_l1": float(np.mean(np.abs(delta))),
        "decoded_temporal_acceleration_l1": float(np.mean(np.abs(acceleration))),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    scalar_keys = [
        key for key, value in rows[0].items()
        if key not in {"case", "task", "mode", "per_latent_frame_mse_to_bid"}
        and isinstance(value, (float, int))
    ]
    result: dict[str, object] = {}
    for key in scalar_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
        }
    frame_rows = np.asarray([row["per_latent_frame_mse_to_bid"] for row in rows])
    result["per_latent_frame_mse_to_bid_mean"] = frame_rows.mean(axis=0).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bid-root", type=Path, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--ar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {"bid": args.bid_root, "causal": args.causal_root, "ar": args.ar_root}
    case_sets = {
        mode: {path.name for path in root.iterdir() if (path / "INFERENCE_COMPLETE").is_file()}
        for mode, root in roots.items()
    }
    if len(case_sets["bid"]) != 20 or any(cases != case_sets["bid"] for cases in case_sets.values()):
        raise ValueError(f"The three modes do not contain the same 20 complete cases: {case_sets}")

    rows: list[dict[str, object]] = []
    for case in sorted(case_sets["bid"]):
        for task in TASKS:
            latents = {
                mode: load_latent(root / case / task / "prediction_latent.pt", task)
                for mode, root in roots.items()
            }
            videos = {
                mode: load_video(root / case / task / "prediction.mp4")
                for mode, root in roots.items()
            }
            for mode in MODES:
                row: dict[str, object] = {"case": case, "task": task, "mode": mode}
                row.update(latent_metrics(latents[mode], latents["bid"]))
                row.update(video_metrics(videos[mode], videos["bid"]))
                rows.append(row)
        print(f"measured {case}", flush=True)

    grouped = {
        mode: {
            task: aggregate([row for row in rows if row["mode"] == mode and row["task"] == task])
            for task in TASKS
        }
        for mode in MODES
    }
    grouped["overall"] = {
        mode: aggregate([row for row in rows if row["mode"] == mode]) for mode in MODES
    }
    report = {
        "schema": "wan22_ditto20_bid_causal_ar_temporal_metrics_v1",
        "warning": "Ditto20 has no task ground truth; bid is a reference, not an accuracy target.",
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "roots": {mode: str(root.resolve()) for mode, root in roots.items()},
        "num_cases": 20,
        "tasks": list(TASKS),
        "aggregate": grouped,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    csv_keys = [key for key in rows[0] if key != "per_latent_frame_mse_to_bid"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_keys} for row in rows)
    print(f"wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
