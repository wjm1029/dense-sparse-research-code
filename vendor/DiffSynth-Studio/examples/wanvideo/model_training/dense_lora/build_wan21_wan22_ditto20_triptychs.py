"""Build matched RGB/Wan2.1/Wan2.2 depth and normal triptychs."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


WAN21_ROOT = Path("/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_ditto20_depth_normal")
WAN22_ROOT = Path("/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_ditto20_depth_normal")
OUTPUT_ROOT = Path("/pfs/mayuema/dense/wan/outputs/wan21_vs_wan22_ditto20_triptychs")
SELECTION = WAN21_ROOT / "selection_manifest.json"


def read_video(path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(path))
    try:
        frames = [np.asarray(frame)[..., :3] for frame in reader]
    finally:
        reader.close()
    video = np.stack(frames).astype(np.uint8, copy=False)
    if video.shape != (81, 480, 832, 3):
        raise ValueError(f"Unexpected video shape {video.shape}: {path}")
    return video


def label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 500), 28), fill=(0, 0, 0))
    draw.text((7, 7), text, fill=(255, 255, 255))
    return np.asarray(image)


def fixed_turbo(video: np.ndarray) -> np.ndarray:
    """Apply one shared byte-domain Turbo map; never normalize per case/frame."""
    gray = np.rint(video.astype(np.float32).mean(axis=-1)).clip(0, 255).astype(np.uint8)
    return np.stack(
        [cv2.cvtColor(cv2.applyColorMap(frame, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB) for frame in gray]
    )


def write_video(path: Path, frames: np.ndarray) -> None:
    writer = imageio.get_writer(str(path), fps=16, quality=7, macro_block_size=16)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def build_triptych(
    path: Path,
    source: np.ndarray,
    wan21: np.ndarray,
    wan22: np.ndarray,
    wan21_label: str,
    wan22_label: str,
) -> None:
    frames = np.stack(
        [
            np.concatenate(
                (
                    label(source[index], "RGB input"),
                    label(wan21[index], wan21_label),
                    label(wan22[index], wan22_label),
                ),
                axis=1,
            )
            for index in range(81)
        ]
    )
    write_video(path, frames)
    Image.fromarray(frames[40]).save(path.with_name(path.stem + "_frame41.png"))


def main() -> None:
    cases = json.loads(SELECTION.read_text(encoding="utf-8"))["cases"]
    if len(cases) != 20:
        raise ValueError(f"Expected 20 cases, got {len(cases)}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        case_name = case["case_name"]
        source = read_video(Path(case["path"]))
        wan21_depth = fixed_turbo(read_video(WAN21_ROOT / case_name / "depth" / "prediction_gray.mp4"))
        wan22_depth = fixed_turbo(read_video(WAN22_ROOT / case_name / "depth" / "prediction.mp4"))
        wan21_normal = read_video(WAN21_ROOT / case_name / "normal" / "prediction.mp4")
        wan22_normal = read_video(WAN22_ROOT / case_name / "normal" / "prediction.mp4")

        case_dir = OUTPUT_ROOT / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        build_triptych(
            case_dir / "depth_rgb_wan21_wan22.mp4",
            source,
            wan21_depth,
            wan22_depth,
            "Wan2.1 depth (fixed Turbo)",
            "Wan2.2 depth (fixed Turbo)",
        )
        build_triptych(
            case_dir / "normal_rgb_wan21_wan22.mp4",
            source,
            wan21_normal,
            wan22_normal,
            "Wan2.1 normal",
            "Wan2.2 normal",
        )
        rows.append({"case_name": case_name, "source": case["path"]})
        print(json.dumps({"completed": case_name}), flush=True)

    metadata = {
        "schema": "wan21-vs-wan22-ditto20-triptychs-v1",
        "cases": rows,
        "layouts": {
            "depth": "RGB input | Wan2.1 depth | Wan2.2 depth",
            "normal": "RGB input | Wan2.1 normal | Wan2.2 normal",
        },
        "depth_visualization": {
            "source": "decoded prediction grayscale video",
            "normalization": "none",
            "mapping": "fixed uint8 0..255 through OpenCV COLORMAP_TURBO",
            "semantic": "higher inverse-depth value uses the high end of Turbo",
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_ROOT / "COMPLETE").write_text(json.dumps({"cases": 20, "triptychs": 40}), encoding="utf-8")


if __name__ == "__main__":
    main()
