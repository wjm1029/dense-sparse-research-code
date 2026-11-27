"""Prepare exact-frame long-video inputs for Wan2.2 AR inference.

The source is first converted to the requested CFR.  A centered crop with the
exact 26:15 target aspect ratio is applied before isotropic scaling to 832x480,
so geometry is never stretched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def probe(path: Path, *, count_frames: bool = False) -> dict[str, object]:
    entries = "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames"
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        command.append("-count_frames")
        entries += ",nb_read_frames"
    command += ["-show_entries", entries, "-of", "json"]
    return json.loads(run(command + [str(path)]))["streams"][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames", type=int, default=477)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    if min(args.frames, args.fps, args.width, args.height) <= 0:
        raise ValueError("frames, fps, width and height must be positive")

    source_info = probe(source)
    source_width = int(source_info["width"])
    source_height = int(source_info["height"])
    # Use the largest integer crop whose ratio exactly equals the output ratio.
    from math import gcd

    divisor = gcd(args.width, args.height)
    ratio_width, ratio_height = args.width // divisor, args.height // divisor
    crop_unit = min(source_width // ratio_width, source_height // ratio_height)
    crop_width, crop_height = ratio_width * crop_unit, ratio_height * crop_unit
    if crop_unit <= 0:
        raise ValueError("Source is too small for the requested aspect ratio")

    duration = float(
        run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(source),
        ]).strip()
    )
    normalized_frames = int(duration * args.fps + 0.5)
    segment_count = normalized_frames // args.frames
    if segment_count == 0:
        raise ValueError("Source is shorter than one requested segment")

    outputs = []
    for index in range(segment_count):
        start, end = index * args.frames, (index + 1) * args.frames
        output = output_dir / f"input_{index:03d}.mp4"
        if output.exists() and not args.overwrite:
            raise FileExistsError(output)
        vf = (
            f"fps={args.fps},"
            f"crop={crop_width}:{crop_height}:(iw-{crop_width})/2:(ih-{crop_height})/2,"
            f"scale={args.width}:{args.height}:flags=lanczos,setsar=1,"
            f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
        )
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if args.overwrite else "-n",
                "-i", str(source), "-an", "-vf", vf, "-vsync", "cfr",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
            ],
            check=True,
        )
        info = probe(output, count_frames=True)
        actual_frames = int(info.get("nb_read_frames") or info.get("nb_frames") or -1)
        if (
            actual_frames != args.frames
            or int(info["width"]) != args.width
            or int(info["height"]) != args.height
            or info["avg_frame_rate"] != f"{args.fps}/1"
        ):
            raise RuntimeError(f"Invalid prepared segment {output}: {info}")
        outputs.append(
            {
                "case_name": output.stem,
                "path": str(output),
                "start_frame_16fps": start,
                "end_frame_16fps_exclusive": end,
                "frames": actual_frames,
            }
        )

    manifest = {
        "schema": "wan22_long_ar_input_segments_v1",
        "source": str(source),
        "source_probe": source_info,
        "fps": args.fps,
        "frames_per_segment": args.frames,
        "output_size": [args.width, args.height],
        "geometry": {
            "operation_order": ["center_crop", "isotropic_scale"],
            "crop_size": [crop_width, crop_height],
            "crop_origin": [
                (source_width - crop_width) // 2,
                (source_height - crop_height) // 2,
            ],
            "aspect_ratio": f"{ratio_width}:{ratio_height}",
            "distortion": False,
        },
        "estimated_normalized_frames": normalized_frames,
        "discarded_tail_frames": normalized_frames - segment_count * args.frames,
        "cases": outputs,
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
