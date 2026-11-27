#!/usr/bin/env python3
"""Build matched RGB/offline/causal-trained comparisons for all Dense4 tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


TASKS = ("depth", "normal", "matte", "raymap")
CANVAS_W = 1920
HEADER_H = 52


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def visualize(frame: np.ndarray, task: str) -> np.ndarray:
    if task == "depth":
        # Fixed mapping across every frame/model; per-frame normalization would
        # hide scale and temporal instability differences.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    if task == "matte":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame


def header(label: str, width: int) -> np.ndarray:
    canvas = np.full((HEADER_H, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        canvas,
        label,
        (16, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72 if width == 480 else 0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def read_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {path}")
    return capture


def make_video(
    *,
    case: str,
    task: str,
    source: Path,
    offline: Path,
    trained: Path,
    vipe: Path | None,
    output_root: Path,
) -> dict:
    paths = [source, offline, trained] + ([vipe] if vipe is not None else [])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{case}/{task}: missing {missing}")
    captures = [read_capture(path) for path in paths]
    fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
    if max(fps_values) - min(fps_values) > 1e-3:
        raise ValueError(f"{case}/{task}: FPS mismatch {fps_values}")
    if len(set(frame_counts)) != 1:
        raise ValueError(f"{case}/{task}: frame-count mismatch {frame_counts}")

    labels = ["Input RGB", f"Offline ({task})", f"Causal trained ({task})"]
    if vipe is not None:
        labels.append("ViPE (depth)")
    panel_w = CANVAS_W // len(labels)
    panel_h = round(panel_w * 9 / 16)
    out_dir = output_root / case / task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison.mp4"
    working_path = out_dir / "comparison.working.mp4"
    writer = cv2.VideoWriter(
        str(working_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_values[0],
        (CANVAS_W, panel_h + HEADER_H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create {working_path}")

    first = None
    middle = None
    written = 0
    while True:
        reads = [capture.read() for capture in captures]
        if not all(ok for ok, _ in reads):
            break
        raw_frames = [frame for _, frame in reads]
        panels = []
        for index, (label, frame) in enumerate(zip(labels, raw_frames)):
            if index == 0 or (vipe is not None and index == len(raw_frames) - 1):
                shown = frame
            else:
                shown = visualize(frame, task)
            panels.append(np.vstack((header(label, panel_w), letterbox(shown, panel_w, panel_h))))
        comparison = np.hstack(panels)
        if first is None:
            first = comparison.copy()
        if written == frame_counts[0] // 2:
            middle = comparison.copy()
        writer.write(comparison)
        written += 1
    for capture in captures:
        capture.release()
    writer.release()
    if written != frame_counts[0]:
        raise RuntimeError(f"{case}/{task}: wrote {written}/{frame_counts[0]} frames")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(working_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        check=True,
    )
    working_path.unlink()
    cv2.imwrite(str(out_dir / "preview_first.png"), first)
    cv2.imwrite(str(out_dir / "preview_middle.png"), middle)
    return {
        "case": case,
        "task": task,
        "source": str(source),
        "offline": str(offline),
        "causal_trained": str(trained),
        "vipe": str(vipe) if vipe is not None else None,
        "output": str(out_path),
        "fps": fps_values[0],
        "frames": written,
        "size": [CANVAS_W, panel_h + HEADER_H],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--trained-root", type=Path, required=True)
    parser.add_argument("--vipe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    case_dirs = sorted(path for path in args.trained_root.iterdir() if (path / "manifest.json").is_file())
    if len(case_dirs) != 20:
        raise RuntimeError(f"Expected 20 trained cases, found {len(case_dirs)}")
    results = []
    for case_index, case_dir in enumerate(case_dirs, 1):
        case = case_dir.name
        manifest = json.loads((case_dir / "manifest.json").read_text())
        source = Path(manifest["source"])
        for task in TASKS:
            vipe = args.vipe_root / case / "vipe_depth_fixed_turbo.mp4" if task == "depth" else None
            result = make_video(
                case=case,
                task=task,
                source=source,
                offline=args.offline_root / case / task / "prediction.mp4",
                trained=args.trained_root / case / task / "prediction.mp4",
                vipe=vipe,
                output_root=args.output_root,
            )
            results.append(result)
        print(f"[{case_index}/{len(case_dirs)}] {case}: {', '.join(TASKS)}", flush=True)

    summary = {
        "schema": "wan22_all_modalities_comparison_v1",
        "layout": {
            "depth": ["input_rgb", "offline", "causal_trained", "vipe"],
            "normal": ["input_rgb", "offline", "causal_trained"],
            "matte": ["input_rgb", "offline", "causal_trained"],
            "raymap": ["input_rgb", "offline", "causal_trained"],
        },
        "visualization": {
            "depth_wan": "fixed uint8 0..255 OpenCV COLORMAP_TURBO",
            "depth_vipe": "precomputed clip-global inverse-depth p2/p98 fixed Turbo",
            "normal": "decoded RGB",
            "matte": "decoded grayscale",
            "raymap": "decoded RGB",
        },
        "items": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "COMPLETE").write_text(json.dumps({"cases": 20, "videos": 80}) + "\n")
    print(f"Wrote 80 comparison videos to {args.output_root}")


if __name__ == "__main__":
    main()
