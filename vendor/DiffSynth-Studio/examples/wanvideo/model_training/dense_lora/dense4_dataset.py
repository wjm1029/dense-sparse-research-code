"""Four-task dense target readers matching wan21_dense_data_integration.md."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

TASK_TO_ID = {"depth": 0, "normal": 1, "matte": 2, "raymap": 3}
TASK_PROMPTS = {
    "depth": "output: depth",
    "normal": "output: surface normal",
    "raymap": "output: camera raymap",
}
SUBJECT_LABELS = (
    ("crow", "crow"), ("crane", "crane"), ("gorilla", "gorilla"),
    ("monkey", "monkey"), ("bear", "bear"), ("horse", "horse"),
    ("lion", "lion"), ("dog", "dog"), ("bird", "bird"),
    ("robot", "robot"), ("person", "person"), ("human", "person"),
    ("humanoid", "person"), ("character", "person"), ("figure", "person"),
    ("child", "person"), ("soldier", "person"), ("dancer", "person"),
    ("creature", "creature"),
)


def subject_label_from_answer(answer: str) -> str:
    first_sentence = answer.split(".", 1)[0]
    match = re.search(
        r"\b(?:a|the) single (.+?)(?: in |,| viewed| positioned| performing| that |$)",
        first_sentence,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Cannot extract subject from Qwen answer: {answer!r}")
    phrase = match.group(1).lower().strip()
    for keyword, label in SUBJECT_LABELS:
        if keyword in phrase:
            return label
    raise ValueError(f"Unsupported subject phrase {phrase!r}")


def load_subject_map(prompt_dir: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    paths = sorted(Path(prompt_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No prompt JSONL files under {prompt_dir}")
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                video = Path(str(row.get("video_path", "")))
                sample_id = f"{video.parent.name}/{video.stem}"
                answer = row.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"Missing answer at {path}:{line_number}")
                if sample_id in result:
                    raise ValueError(f"Duplicate prompt for {sample_id}")
                result[sample_id] = subject_label_from_answer(" ".join(answer.split()))
    return result


def _read_normal_exr(path: Path) -> np.ndarray:
    try:
        import OpenEXR  # type: ignore
    except ImportError:
        import cv2

        packed = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if packed is None or packed.ndim != 3 or packed.shape[-1] < 3:
            raise ValueError(f"OpenCV failed to read RGB normal EXR: {path}")
        normal = np.asarray(packed[..., :3][..., ::-1], dtype=np.float32)
    else:
        channels = OpenEXR.File(str(path)).channels()
        if "RGBA" in channels:
            normal = np.asarray(channels["RGBA"].pixels[..., :3], dtype=np.float32)
        elif all(channel in channels for channel in ("R", "G", "B")):
            normal = np.stack(
                [channels[channel].pixels for channel in ("R", "G", "B")], axis=-1
            ).astype(np.float32)
        else:
            raise ValueError(f"Normal EXR has no RGB(A) channels: {path}")
    if normal.ndim != 3 or normal.shape[-1] != 3 or not np.isfinite(normal).all():
        raise ValueError(f"Invalid normal EXR {normal.shape}: {path}")
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    valid = norm > 1.0e-6
    return np.where(valid, normal / np.maximum(norm, 1.0e-6), 0.0).astype(np.float32)


def _read_normal_png(path: Path) -> np.ndarray:
    """Read RGB uint8 normals mapped from signed unit vectors."""
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB normal PNG, got {image.mode}: {path}")
        packed = np.asarray(image, dtype=np.uint8)
    normal = packed.astype(np.float32) / 127.5 - 1.0
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    if not np.isfinite(normal).all() or np.any(norm <= 1.0e-6):
        raise ValueError(f"Invalid encoded normal PNG: {path}")
    return (normal / norm).astype(np.float32)


def _read_normal(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        return _read_normal_exr(path)
    if path.suffix.lower() == ".png":
        return _read_normal_png(path)
    raise ValueError(f"Unsupported normal format: {path}")


def _read_matte(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mode = image.mode
        value = np.asarray(image)
    if value.ndim == 3:
        value = value[..., 0]
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"Expected integer matte PNG, got {value.dtype}: {path}")
    scale = 65535.0 if "16" in mode or int(value.max()) > 255 else 255.0
    return value.astype(np.float32) / scale


def _read_raymap(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB raymap, got {image.mode}: {path}")
        value = np.asarray(image, dtype=np.uint8)
    return value.astype(np.float32) / 255.0


class Dense4TargetDataset(Dataset):
    """Read normal/matte/raymap targets for clips in the validated depth index."""

    target_tasks = ("normal", "matte", "raymap")

    def __init__(
        self,
        dataset_root: str | Path,
        depth_index_path: str | Path,
        prompt_dir: str | Path,
        *,
        max_samples: int = 0,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        payload = json.loads(Path(depth_index_path).read_text(encoding="utf-8"))
        if payload.get("schema") != "wan21-depth-dataset-index-v1":
            raise ValueError("Unsupported source depth index")
        samples = list(payload["samples"])
        if max_samples > 0:
            samples = samples[:max_samples]
        subjects = load_subject_map(prompt_dir)
        missing = [row["id"] for row in samples if row["id"] not in subjects]
        if missing:
            raise ValueError(f"Matte subject coverage missing {len(missing)} clips: {missing[:5]}")
        self.samples = [
            {
                "id": row["id"],
                "clip_dir": self.root / row["id"],
                "rgb_mean": float(row["rgb_mean"]),
                "subject_label": subjects[row["id"]],
            }
            for row in samples
        ]
        if not self.samples:
            raise ValueError("No dense4 target samples")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def prompt_for(task: str, subject_label: str) -> str:
        if task == "matte":
            return f"output: {subject_label} alpha matte"
        return TASK_PROMPTS[task]

    def _paths(self, clip_dir: Path, task: str) -> list[Path]:
        if task == "normal":
            directory = clip_dir / "normal"
            if (directory / "000001.exr").is_file():
                suffix = ".exr"
            elif (directory / "000001.png").is_file():
                marker = clip_dir / "NORMAL_DONE"
                if not marker.is_file():
                    raise FileNotFoundError(f"Missing complete normal marker: {marker}")
                payload = json.loads(marker.read_text(encoding="utf-8"))
                if payload.get("schema") != "lingbot-wan-normal81-v2":
                    raise ValueError(f"Unsupported normal marker: {marker}")
                suffix = ".png"
            else:
                raise FileNotFoundError(f"No supported normal frames under {directory}")
        elif task == "matte":
            directory, suffix = clip_dir / "matte", ".png"
        elif task == "raymap":
            directory, suffix = clip_dir / "genception" / "camera_raymap", ".png"
            if not (directory / ".png_complete.json").is_file():
                raise FileNotFoundError(f"Missing complete raymap marker: {directory}")
        else:
            raise ValueError(f"Unsupported target task {task!r}")
        return [directory / f"{frame:06d}{suffix}" for frame in range(1, 82)]

    def load_target(self, row: dict, task: str) -> torch.Tensor:
        frames = []
        for path in self._paths(row["clip_dir"], task):
            if task == "normal":
                value = _read_normal(path)
            elif task == "matte":
                value = _read_matte(path)
            else:
                value = _read_raymap(path)
            if value.shape[:2] != (480, 832) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {task} frame {value.shape}: {path}")
            frames.append(value)
        target = torch.from_numpy(np.stack(frames))
        if task in ("normal", "raymap"):
            target = target.permute(0, 3, 1, 2)
        else:
            target = target[:, None].repeat(1, 3, 1, 1)
        if task != "normal":
            target = target.mul_(2.0).sub_(1.0)
        if target.shape != (81, 3, 480, 832) or not torch.isfinite(target).all():
            raise ValueError(f"Invalid {task} video tensor for {row['id']}")
        return target

    def __getitem__(self, index: int) -> dict:
        row = self.samples[index]
        return {
            "clip_id": row["id"],
            "subject_label": row["subject_label"],
            "rgb_mean": row["rgb_mean"],
            "targets": {task: self.load_target(row, task) for task in self.target_tasks},
            "prompts": {
                task: self.prompt_for(task, row["subject_label"])
                for task in self.target_tasks
            },
        }
