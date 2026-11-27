"""Build matched long-video comparisons for AR self-forcing with/without OPSD."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


EXPECTED_CASES = tuple(f"input_{index:03d}" for index in range(6))
EXPECTED_FRAMES = 477
EXPECTED_SIZE = (832, 480)
EXPECTED_FPS = 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_manifest", type=Path, required=True)
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--opsd_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path, *, size: tuple[int, int] = EXPECTED_SIZE) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,avg_frame_rate",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    value = {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "frames": int(stream["nb_frames"]), "fps": numerator / denominator,
    }
    expected = {
        "width": size[0], "height": size[1],
        "frames": EXPECTED_FRAMES, "fps": EXPECTED_FPS,
    }
    if value != expected:
        raise ValueError(f"Unexpected video contract for {path}: {value}")
    return value


def labelled(name: str, index: int) -> str:
    safe = name.replace("'", "\\'")
    return (
        f"[{index}:v]drawtext=text='{safe}':x=18:y=18:fontsize=30:"
        "fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=8"
    )


def build_video(inputs: list[Path], output: Path) -> None:
    filters = [
        labelled("RGB input", 0) + "[rgb]",
        labelled("Depth - no OPSD", 1) + "[d0]",
        labelled("Depth + OPSD", 2) + "[d1]",
        labelled("Normal - no OPSD", 3) + "[n0]",
        labelled("Normal + OPSD", 4) + "[n1]",
        "[rgb]split=2[rgb_top][rgb_bottom]",
        "[rgb_top][d0][d1]hstack=inputs=3[top]",
        "[rgb_bottom][n0][n1]hstack=inputs=3[bottom]",
        "[top][bottom]vstack=inputs=2[out]",
    ]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[out]",
        "-an", "-r", "16", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ])
    subprocess.run(command, check=True)


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    reader = imageio.get_reader(str(path))
    try:
        return np.asarray(reader.get_data(frame_index))[..., :3]
    finally:
        reader.close()


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 260, 42), fill=(0, 0, 0))
    draw.text((16, 16), text, fill=(255, 255, 255))
    return np.asarray(image)


def build_timeline(inputs: list[Path], output: Path) -> None:
    indices = (0, 119, 238, 357, 476)
    labels = ("RGB", "Depth no OPSD", "Depth OPSD", "Normal no OPSD", "Normal OPSD")
    rows = []
    for path, label in zip(inputs, labels):
        frames = [add_label(read_frame(path, index), f"{label}  f={index}") for index in indices]
        rows.append(np.concatenate(frames, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(output)


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    case_names = tuple(case["case_name"] for case in selection["cases"])
    if case_names != EXPECTED_CASES:
        raise ValueError(f"Unexpected case order: {case_names}")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing non-empty output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for case in selection["cases"]:
        name = case["case_name"]
        inputs = [
            Path(case["path"]),
            args.baseline_root / name / "depth" / "prediction.mp4",
            args.opsd_root / name / "depth" / "prediction.mp4",
            args.baseline_root / name / "normal" / "prediction.mp4",
            args.opsd_root / name / "normal" / "prediction.mp4",
        ]
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
            probe(path)
        video = args.output_root / f"{name}_opsd_comparison.mp4"
        timeline = args.output_root / f"{name}_timeline.png"
        build_video(inputs, video)
        build_timeline(inputs, timeline)
        probe_result = probe(video, size=(2496, 960))
        results.append({
            "case": name,
            "video": str(video), "video_sha256": sha256(video),
            "timeline": str(timeline), "timeline_sha256": sha256(timeline),
        })
        print(json.dumps(results[-1]), flush=True)

    baseline_metadata = json.loads((args.baseline_root / EXPECTED_CASES[0] / "manifest.json").read_text())
    opsd_metadata = json.loads((args.opsd_root / EXPECTED_CASES[0] / "manifest.json").read_text())
    manifest = {
        "schema": "wan22_opsd_long477_visual_comparison_v1",
        "layout": [["rgb", "depth_no_opsd", "depth_opsd"], ["rgb", "normal_no_opsd", "normal_opsd"]],
        "selection_manifest": str(args.selection_manifest.resolve()),
        "baseline_checkpoint": baseline_metadata["checkpoint"],
        "baseline_checkpoint_sha256": baseline_metadata["checkpoint_sha256"],
        "opsd_checkpoint": opsd_metadata["checkpoint"],
        "opsd_checkpoint_sha256": opsd_metadata["checkpoint_sha256"],
        "cases": results,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_root / "COMPLETE").write_text(
        json.dumps({"cases": len(results), "videos": len(results)}) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
