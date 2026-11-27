"""Depth-only paired video dataset for GenCeption-style Wan2.1 training.

The implementation intentionally mirrors the verified production preprocessing in
``/pfs/mayuema/dense/wan/docs/wan21_dense_data_integration.md``.  It is kept
standalone from the generic DiffSynth dataset because source RGB and target depth
have different readers but must stay pixel- and frame-aligned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# OpenCV ships with EXR support disabled unless this flag is set before import.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _sorted_frames(directory: Path, suffix: str) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == suffix
    )


def _clip_rgb_mean_from_quality(clip_dir: Path, expected_frames: int) -> float:
    path = clip_dir / "quality.json"
    if not path.is_file():
        raise FileNotFoundError(f"Brightness filtering requires {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, dict) or len(frames) != expected_frames:
        raise ValueError(f"{path}: expected {expected_frames} frame records")
    try:
        frame_ids = {int(frame_id) for frame_id in frames}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: frame ids must be integers") from error
    expected_ids = set(range(1, expected_frames + 1))
    if frame_ids != expected_ids:
        raise ValueError(f"{path}: frame ids are not exactly 1..{expected_frames}")
    values = []
    for frame_id, frame in frames.items():
        rgb = frame.get("rgb") if isinstance(frame, dict) else None
        if not isinstance(rgb, dict) or "mean" not in rgb:
            raise ValueError(f"{path}: frame {frame_id!r} is missing rgb.mean")
        values.append(float(rgb["mean"]))
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: rgb.mean contains NaN/Inf")
    return float(np.mean(values))


def _read_depth_exr(path: Path) -> np.ndarray:
    """Read the production packed RGBA EXR's B channel.

    OpenEXR exposes packed channels as RGBA, where depth is index 2.  OpenCV
    exposes the same pixels as BGRA, so depth is index 0.  Prefer OpenEXR when
    installed and use the explicit OpenCV channel-order fallback otherwise.
    """

    try:
        import OpenEXR  # type: ignore
    except ImportError:
        import cv2

        packed = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if packed is None:
            raise ValueError(f"OpenCV failed to read depth EXR: {path}")
        packed = np.asarray(packed)
        if packed.ndim == 2:
            depth = packed
        elif packed.ndim == 3 and packed.shape[-1] >= 3:
            depth = packed[..., 0]  # BGRA/BGR: B is index 0.
        else:
            raise ValueError(f"Unexpected OpenCV EXR shape {packed.shape}: {path}")
    else:
        channels = OpenEXR.File(str(path)).channels()
        if "RGBA" in channels:
            packed = np.asarray(channels["RGBA"].pixels)
            if packed.ndim != 3 or packed.shape[-1] < 3:
                raise ValueError(f"Unexpected packed depth shape {packed.shape}: {path}")
            depth = packed[..., 2]
        elif "B" in channels:
            depth = channels["B"].pixels
        elif "Y" in channels:
            depth = channels["Y"].pixels
        elif len(channels) == 1:
            depth = next(iter(channels.values())).pixels
        else:
            raise ValueError(f"Depth EXR has no unambiguous depth channel: {path}")

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got {depth.shape}: {path}")
    return depth


def _read_normalized_depth_png(path: Path) -> np.ndarray:
    """Read offline clip-normalized inverse depth stored as uint16 [0, 1]."""
    import cv2

    packed = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if packed is None:
        raise ValueError(f"OpenCV failed to read normalized depth PNG: {path}")
    if packed.dtype != np.uint16 or packed.ndim != 2:
        raise ValueError(
            f"Expected uint16 HxW normalized depth PNG, got {packed.dtype}/{packed.shape}: {path}"
        )
    return packed.astype(np.float32) / np.float32(65535.0)


def _ray_distance_to_camera_z(depth: np.ndarray, intrinsic: list) -> np.ndarray:
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid camera intrinsic matrix: {intrinsic!r}")
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Camera focal lengths must be positive: fx={fx}, fy={fy}")
    height, width = depth.shape
    pixel_x = np.arange(width, dtype=np.float32) + 0.5
    pixel_y = np.arange(height, dtype=np.float32) + 0.5
    ray_length = np.sqrt(
        1.0
        + np.square((pixel_x[None, :] - cx) / fx)
        + np.square((pixel_y[:, None] - cy) / fy)
    )
    return depth / ray_length


class DepthVideoDataset(Dataset):
    """Load aligned RGB and clip-normalized inverse-camera-Z videos."""

    prompt = "output: depth"

    def __init__(
        self,
        root: str | Path,
        *,
        num_frames: int = 477,
        height: int = 480,
        width: int = 832,
        min_rgb_mean: float = 40.0,
        overfit_clip: str | None = None,
        max_samples: int = 0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.min_rgb_mean = float(min_rgb_mean)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        if self.num_frames != 477 or (self.height, self.width) != (480, 832):
            raise ValueError("This implementation is fixed to 477 frames at 832x480")

        if overfit_clip:
            candidates = [self.root / overfit_clip]
        else:
            candidates = sorted(self.root.glob("*/*"))

        samples: list[dict] = []
        stats = {
            "scanned": 0,
            "missing_done": 0,
            "missing_modality": 0,
            "frame_count": 0,
            "frame_mismatch": 0,
            "dark": 0,
            "unsupported_storage": 0,
        }
        for clip_dir in candidates:
            stats["scanned"] += 1
            if not clip_dir.is_dir() or not (clip_dir / "DONE").is_file():
                stats["missing_done"] += 1
                continue
            rgb_dir, depth_dir = clip_dir / "rgb", clip_dir / "depth"
            if not rgb_dir.is_dir() or not depth_dir.is_dir():
                stats["missing_modality"] += 1
                continue
            rgb_mean = _clip_rgb_mean_from_quality(clip_dir, self.num_frames)
            if rgb_mean < self.min_rgb_mean:
                stats["dark"] += 1
                continue
            # The production integration contract fixes both modalities to
            # aligned 000001..000081 names.  Constructing them avoids two GPFS
            # directory scans per clip; __getitem__ still decodes every file.
            frame_names = [f"{frame_id:06d}" for frame_id in range(1, self.num_frames + 1)]
            if (rgb_dir / "000001.png").is_file() and (depth_dir / "000001.exr").is_file():
                rgb_paths = [rgb_dir / f"{name}.png" for name in frame_names]
                depth_paths = [depth_dir / f"{name}.exr" for name in frame_names]
            elif (rgb_dir / "000001.jpg").is_file() and (depth_dir / "000001.png").is_file():
                rgb_paths = [rgb_dir / f"{name}.jpg" for name in frame_names]
                depth_paths = [depth_dir / f"{name}.png" for name in frame_names]
            else:
                stats["unsupported_storage"] += 1
                continue
            sample_id = f"{clip_dir.parent.name}/{clip_dir.name}"
            samples.append(
                {
                    "id": sample_id,
                    "clip_dir": clip_dir,
                    "rgb_paths": rgb_paths,
                    "depth_paths": depth_paths,
                    "rgb_mean": rgb_mean,
                }
            )
            if max_samples > 0 and len(samples) >= max_samples:
                break

        if not samples:
            raise ValueError(
                f"No valid depth clips found under {self.root}; scan statistics: {stats}"
            )
        self.samples = samples
        self.scan_stats = stats

    def index_payload(self) -> dict:
        """Return a compact, portable index so DDP ranks do not rescan GPFS."""
        return {
            "schema": "wan22-depth477-dataset-index-v1",
            "root": str(self.root),
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "min_rgb_mean": self.min_rgb_mean,
            "scan_stats": self.scan_stats,
            "samples": [
                {
                    "id": row["id"],
                    "rgb_mean": row["rgb_mean"],
                    "rgb_names": [path.name for path in row["rgb_paths"]],
                    "depth_names": [path.name for path in row["depth_paths"]],
                }
                for row in self.samples
            ],
        }

    @classmethod
    def from_index_payload(
        cls,
        payload: dict,
        *,
        root: str | Path,
        min_rgb_mean: float,
    ) -> "DepthVideoDataset":
        """Reconstruct an already validated dataset index without directory scans."""
        resolved_root = Path(root).expanduser().resolve()
        if payload.get("schema") != "wan22-depth477-dataset-index-v1":
            raise ValueError("Unsupported depth dataset index")
        if Path(payload.get("root", "")).resolve() != resolved_root:
            raise ValueError("Depth dataset index root mismatch")
        if float(payload.get("min_rgb_mean", -1)) != float(min_rgb_mean):
            raise ValueError("Depth dataset index brightness threshold mismatch")

        dataset = cls.__new__(cls)
        dataset.root = resolved_root
        dataset.num_frames = int(payload["num_frames"])
        dataset.height = int(payload["height"])
        dataset.width = int(payload["width"])
        dataset.min_rgb_mean = float(payload["min_rgb_mean"])
        dataset.scan_stats = payload["scan_stats"]
        dataset.samples = []
        for item in payload["samples"]:
            clip_dir = resolved_root / item["id"]
            dataset.samples.append(
                {
                    "id": item["id"],
                    "clip_dir": clip_dir,
                    "rgb_paths": [clip_dir / "rgb" / name for name in item["rgb_names"]],
                    "depth_paths": [clip_dir / "depth" / name for name in item["depth_names"]],
                    "rgb_mean": float(item["rgb_mean"]),
                }
            )
        if not dataset.samples:
            raise ValueError("Depth dataset index contains no samples")
        return dataset

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        row = self.samples[index]
        metadata_path = row["clip_dir"] / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        depth_stats = metadata.get("relative_inverse_depth_preprocess")
        if not isinstance(depth_stats, dict):
            raise ValueError(f"{metadata_path}: missing relative_inverse_depth_preprocess")
        if (
            depth_stats.get("mode") != "inverse_depth_clip_percentile"
            or depth_stats.get("domain") != "inverse_camera_z"
        ):
            raise ValueError(f"{metadata_path}: unsupported depth preprocessing metadata")
        p2, p98 = float(depth_stats.get("p2", 0.0)), float(depth_stats.get("p98", 0.0))
        if not np.isfinite([p2, p98]).all() or not 0.0 < p2 < p98:
            raise ValueError(f"{metadata_path}: invalid inverse-depth span {p2}..{p98}")

        frames_data = metadata.get("frames_data")
        if not isinstance(frames_data, list) or len(frames_data) != self.num_frames:
            raise ValueError(f"{metadata_path}: expected {self.num_frames} frame records")
        frame_records = {int(record["frame"]): record for record in frames_data}
        coordinates = metadata.get("coordinates")
        depth_type = coordinates.get("depth_type") if isinstance(coordinates, dict) else None
        if depth_type not in ("camera_z_meters", "blender_z_ray_distance_meters"):
            raise ValueError(f"{metadata_path}: unsupported depth type {depth_type!r}")

        source_frames: list[np.ndarray] = []
        target_frames: list[np.ndarray] = []
        for rgb_path, depth_path in zip(row["rgb_paths"], row["depth_paths"]):
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape != (self.height, self.width, 3):
                raise ValueError(f"Unexpected RGB shape {rgb.shape}: {rgb_path}")

            if depth_path.suffix.lower() == ".exr":
                depth = _read_depth_exr(depth_path)
                depth_is_normalized = False
            elif depth_path.suffix.lower() == ".png":
                if depth_stats.get("storage") != "uint16_depth01":
                    raise ValueError(
                        f"{metadata_path}: normalized PNG requires storage='uint16_depth01'"
                    )
                depth = _read_normalized_depth_png(depth_path)
                depth_is_normalized = True
            else:
                raise ValueError(f"Unsupported depth file type: {depth_path}")
            if depth.shape != (self.height, self.width):
                raise ValueError(f"Unexpected depth shape {depth.shape}: {depth_path}")
            frame_id = int(depth_path.stem)
            record = frame_records.get(frame_id)
            if record is None:
                raise ValueError(f"{metadata_path}: no frames_data record for frame {frame_id}")
            if depth_is_normalized:
                if not np.isfinite(depth).all() or np.any((depth < 0.0) | (depth > 1.0)):
                    raise ValueError(f"Normalized depth is outside [0,1]: {depth_path}")
                depth01 = depth
            else:
                if depth_type == "blender_z_ray_distance_meters":
                    depth = _ray_distance_to_camera_z(depth, record.get("K"))
                if not np.isfinite(depth).all() or not np.all(depth > 0.0):
                    raise ValueError(f"Depth contains invalid/non-positive camera-Z: {depth_path}")
                inverse_depth = np.reciprocal(depth)
                depth01 = np.clip((inverse_depth - p2) / (p98 - p2), 0.0, 1.0)

            source_frames.append(rgb)
            target_frames.append(depth01.astype(np.float32))

        source = (
            torch.from_numpy(np.stack(source_frames))
            .permute(0, 3, 1, 2)
            .float()
            .div_(127.5)
            .sub_(1.0)
        )
        target = torch.from_numpy(np.stack(target_frames))[:, None].repeat(1, 3, 1, 1)
        target = target.mul_(2.0).sub_(1.0)
        if not torch.isfinite(source).all() or not torch.isfinite(target).all():
            raise ValueError(f"{row['id']}: preprocessed tensor contains NaN/Inf")
        if source.shape != (477, 3, 480, 832) or target.shape != source.shape:
            raise ValueError(
                f"{row['id']}: unexpected source/target shapes {source.shape}/{target.shape}"
            )
        return {
            "source_video": source,
            "target_video": target,
            "prompt": self.prompt,
            "clip_id": row["id"],
            "rgb_mean": row["rgb_mean"],
            "depth_p2": p2,
            "depth_p98": p98,
        }
