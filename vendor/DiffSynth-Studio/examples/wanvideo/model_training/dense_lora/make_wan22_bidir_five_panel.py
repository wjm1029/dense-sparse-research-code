#!/usr/bin/env python3
"""Compose RGB and four Wan2.2 Dense4 predictions into five-panel videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
from matplotlib import colormaps
import numpy as np


TASKS = ("depth", "normal", "matte", "raymap")
PANEL_WIDTH = 384
PANEL_HEIGHT = 222
HEADER_HEIGHT = 50
OUTPUT_FPS = 16.0


def header(label: str) -> np.ndarray:
    image = np.full((HEADER_HEIGHT, PANEL_WIDTH, 3), 24, dtype=np.uint8)
    cv2.putText(
        image,
        label,
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def resize(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)


def depth_range(path: Path) -> tuple[float, float]:
    """Compute one robust range for the full clip to avoid framewise flicker."""
    capture = open_capture(path)
    histogram = np.zeros(256, dtype=np.int64)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram += np.bincount(gray.ravel(), minlength=256)
    capture.release()
    cumulative = np.cumsum(histogram)
    if cumulative[-1] == 0:
        raise ValueError(f"Empty depth video: {path}")
    low = float(np.searchsorted(cumulative, cumulative[-1] * 0.02))
    high = float(np.searchsorted(cumulative, cumulative[-1] * 0.98))
    return low, max(high, low + 1.0)


def visualize(
    frame: np.ndarray, task: str, depth_limits: tuple[float, float]
) -> np.ndarray:
    if task == "depth":
        # Match the Depth Anything-style Spectral_r rendering in the reference
        # image. The robust range is fixed per clip, not recomputed per frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        low, high = depth_limits
        normalized = np.clip((gray.astype(np.float32) - low) / (high - low), 0.0, 1.0)
        rgb = (colormaps["Spectral_r"](normalized)[..., :3] * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if task == "matte":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame


def open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {path}")
    return capture


def compose_case(case_dir: Path, output_root: Path) -> dict:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    source = Path(manifest["source"])
    paths = [source] + [case_dir / task / "prediction.mp4" for task in TASKS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{case_dir.name}: missing {missing}")

    captures = [open_capture(path) for path in paths]
    depth_limits = depth_range(paths[1])
    fps_values = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
    counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
    if max(fps_values) - min(fps_values) > 1e-3:
        raise ValueError(f"{case_dir.name}: FPS mismatch {fps_values}")
    if len(set(counts)) != 1 or counts[0] != 81:
        raise ValueError(f"{case_dir.name}: expected five 81-frame videos, got {counts}")

    output_dir = output_root / case_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "rgb_depth_normal_matte_raymap.mp4"
    working = output_dir / "rgb_depth_normal_matte_raymap.working.mp4"
    labels = ("RGB input", "Depth (Spectral_r)", "Normal", "Matte", "Raymap")
    writer = cv2.VideoWriter(
        str(working),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (PANEL_WIDTH * 5, PANEL_HEIGHT + HEADER_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create {working}")

    previews: dict[int, np.ndarray] = {}
    written = 0
    while True:
        reads = [capture.read() for capture in captures]
        if not all(ok for ok, _ in reads):
            break
        frames = [frame for _, frame in reads]
        shown = [frames[0]] + [
            visualize(frame, task, depth_limits) for frame, task in zip(frames[1:], TASKS)
        ]
        panels = [
            np.vstack((header(label), resize(frame))) for label, frame in zip(labels, shown)
        ]
        canvas = np.hstack(panels)
        if written in (0, 40, 80):
            previews[written] = canvas.copy()
        writer.write(canvas)
        written += 1

    for capture in captures:
        capture.release()
    writer.release()
    if written != 81:
        raise RuntimeError(f"{case_dir.name}: wrote {written}/81 frames")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(working),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )
    working.unlink()
    for frame_index, preview in previews.items():
        cv2.imwrite(str(output_dir / f"preview_{frame_index:03d}.jpg"), preview)
    return {
        "case": case_dir.name,
        "source": str(source),
        "output": str(output),
        "input_fps": fps_values[0],
        "fps": OUTPUT_FPS,
        "frames": written,
        "depth_percentile_range": list(depth_limits),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    cases = sorted(
        path for path in args.prediction_root.iterdir() if (path / "INFERENCE_COMPLETE").is_file()
    )
    if not cases:
        raise RuntimeError(f"No completed cases found in {args.prediction_root}")
    results = []
    for index, case_dir in enumerate(cases, 1):
        results.append(compose_case(case_dir, args.output_root))
        print(f"[{index}/{len(cases)}] {case_dir.name}", flush=True)
    summary = {
        "schema": "wan22_bidir_five_panel_v1",
        "layout": ["input_rgb", "depth_spectral_r", "normal", "matte", "raymap"],
        "depth_visualization": "clip-global uint8 p2..p98 normalization plus matplotlib Spectral_r",
        "cases": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_root / "COMPLETE").write_text(
        json.dumps({"cases": len(results), "videos": len(results)}) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
