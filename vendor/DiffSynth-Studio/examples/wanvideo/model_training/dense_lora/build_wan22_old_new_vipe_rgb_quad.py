"""Build matched old/new Wan2.2, ViPE, and RGB depth comparison videos."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


SELECTION = Path("/pfs/mayuema/dense/wan/outputs/dense4_hyperlora_4gpu_10k_ditto20_depth_normal/selection_manifest.json")
OLD_ROOT = Path("/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_ditto20_depth_normal")
NEW_ROOT = Path("/pfs/mayuema/dense/wan/outputs/wan22_lingbot_stage2_20k_ditto20_depth")
VIPE_ROOT = Path("/pfs/mayuema/dense/wan/outputs/wan21_wan22_vipe_ditto20_depth")
OUTPUT_ROOT = Path("/pfs/mayuema/dense/wan/outputs/wan22_old_new_vipe_rgb_ditto20_depth_quad")


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


def turbo_from_gray_video(video: np.ndarray) -> np.ndarray:
    gray = np.rint(video.astype(np.float32).mean(axis=-1)).clip(0, 255).astype(np.uint8)
    return np.stack([
        cv2.cvtColor(cv2.applyColorMap(frame, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        for frame in gray
    ])


def label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 650), 28), fill=(0, 0, 0))
    draw.text((7, 7), text, fill=(255, 255, 255))
    return np.asarray(image)


def write_video(path: Path, frames: np.ndarray) -> None:
    writer = imageio.get_writer(str(path), fps=16, quality=7, macro_block_size=16)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def main() -> None:
    cases = json.loads(SELECTION.read_text(encoding="utf-8"))["cases"]
    if len(cases) != 20:
        raise ValueError(f"Expected 20 cases, got {len(cases)}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        name = case["case_name"]
        rgb = read_video(Path(case["path"]))
        old = turbo_from_gray_video(read_video(OLD_ROOT / name / "depth" / "prediction.mp4"))
        new = turbo_from_gray_video(read_video(NEW_ROOT / name / "depth" / "prediction.mp4"))
        vipe = read_video(VIPE_ROOT / name / "vipe_depth_fixed_turbo.mp4")
        quad = np.stack([
            np.concatenate((
                label(old[i], "Wan2.2 old depth (fixed Turbo)"),
                label(new[i], "Wan2.2 new stage2-20k depth (fixed Turbo)"),
                label(vipe[i], "ViPE depth (same inverse-depth rule)"),
                label(rgb[i], "RGB input"),
            ), axis=1)
            for i in range(81)
        ])
        case_dir = OUTPUT_ROOT / name
        case_dir.mkdir(parents=True, exist_ok=True)
        write_video(case_dir / "wan22_old_new_vipe_rgb_depth.mp4", quad)
        Image.fromarray(quad[40]).save(case_dir / "wan22_old_new_vipe_rgb_depth_frame41.png")
        rows.append({"case_name": name, "source": case["path"]})
        print(json.dumps({"completed": name}), flush=True)

    manifest = {
        "schema": "wan22-old-new-vipe-rgb-ditto20-depth-quad-v1",
        "layout": "Wan2.2 old | Wan2.2 new stage2-20k | ViPE | RGB",
        "new_checkpoint": "/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_lingbot_stage2_20k/step-20000.safetensors",
        "frames": 81,
        "fps": 16,
        "visualization": "Wan predictions use identical fixed uint8-to-Turbo mapping; ViPE reuses the previously verified clip-global inverse-depth p2/p98 fixed-Turbo output.",
        "cases": rows,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "COMPLETE").write_text(json.dumps({"cases": 20}), encoding="utf-8")


if __name__ == "__main__":
    main()
