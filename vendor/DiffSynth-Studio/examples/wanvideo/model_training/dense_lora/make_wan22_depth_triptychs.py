#!/usr/bin/env python3
"""Build RGB / offline depth / causal-trained depth comparison videos."""

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


PANEL_W = 640
PANEL_H = 360
HEADER_H = 52


def letterbox(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(PANEL_W / w, PANEL_H / h)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    y = (PANEL_H - resized.shape[0]) // 2
    x = (PANEL_W - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def visualize_depth(frame: np.ndarray) -> np.ndarray:
    """Apply one fixed 0..255 Turbo map; never normalize frames independently."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def header(label: str) -> np.ndarray:
    canvas = np.full((HEADER_H, PANEL_W, 3), 24, dtype=np.uint8)
    cv2.putText(
        canvas,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def compose(rgb: np.ndarray, offline: np.ndarray, trained: np.ndarray) -> np.ndarray:
    panels = [
        np.vstack((header("Input RGB"), letterbox(rgb))),
        np.vstack((header("Offline (depth, Turbo)"), letterbox(visualize_depth(offline)))),
        np.vstack((header("Causal trained (depth, Turbo)"), letterbox(visualize_depth(trained)))),
    ]
    return np.hstack(panels)


def make_one(case_dir: Path, offline_root: Path, trained_root: Path, output_root: Path) -> dict:
    case = case_dir.name
    manifest = json.loads((case_dir / "manifest.json").read_text())
    source = Path(manifest["source"])
    offline_path = offline_root / case / "depth" / "prediction.mp4"
    trained_path = trained_root / case / "depth" / "prediction.mp4"
    paths = [source, offline_path, trained_path]
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"{case}: missing {missing}")

    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(cap.isOpened() for cap in captures):
        raise RuntimeError(f"{case}: failed to open one of {paths}")
    fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
    if max(fps_values) - min(fps_values) > 1e-3:
        raise ValueError(f"{case}: FPS mismatch {fps_values}")
    if len(set(frame_counts)) != 1:
        raise ValueError(f"{case}: frame-count mismatch {frame_counts}")

    out_dir = output_root / case
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rgb_offline_causal_trained_depth.mp4"
    working_path = out_dir / "rgb_offline_causal_trained_depth.working.mp4"
    writer = cv2.VideoWriter(
        str(working_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_values[0],
        (PANEL_W * 3, PANEL_H + HEADER_H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"{case}: failed to create {working_path}")

    first_frame = None
    written = 0
    while True:
        frames = [cap.read() for cap in captures]
        if not all(ok for ok, _ in frames):
            break
        triptych = compose(*(frame for _, frame in frames))
        if first_frame is None:
            first_frame = triptych.copy()
        writer.write(triptych)
        written += 1
    for cap in captures:
        cap.release()
    writer.release()
    if written != frame_counts[0]:
        raise RuntimeError(f"{case}: wrote {written}/{frame_counts[0]} frames")
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
    cv2.imwrite(str(out_dir / "preview.png"), first_frame)
    return {
        "case": case,
        "source": str(source),
        "offline": str(offline_path),
        "causal_trained": str(trained_path),
        "output": str(out_path),
        "fps": fps_values[0],
        "frames": written,
        "depth_visualization": "OpenCV COLORMAP_TURBO with fixed uint8 0..255 mapping",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--trained-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    cases = sorted(path for path in args.trained_root.iterdir() if (path / "manifest.json").is_file())
    if not cases:
        raise RuntimeError(f"No completed cases found in {args.trained_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, case_dir in enumerate(cases, 1):
        result = make_one(case_dir, args.offline_root, args.trained_root, args.output_root)
        results.append(result)
        print(f"[{index}/{len(cases)}] {result['case']}: {result['frames']} frames")
    summary = {
        "schema": "wan22_depth_triptych_v1",
        "layout": ["input_rgb", "offline_depth", "causal_trained_depth"],
        "cases": results,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(results)} triptychs to {args.output_root}")


if __name__ == "__main__":
    main()
