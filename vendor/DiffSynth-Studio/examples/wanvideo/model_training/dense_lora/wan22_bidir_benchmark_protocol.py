"""Aspect-preserving preprocessing and dense metrics for Wan2.2 benchmarks.

This is deliberately separate from the official model code.  The protocol is
ours rather than the GenCeption paper's direct-resize protocol: every image is
center-cropped to 26:15, resized to 832x480, and repeated for 81 frames.
Predictions are resized back to the cropped ground-truth resolution before
metrics are computed, so ground truth is never spatially interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


TARGET_WIDTH = 832
TARGET_HEIGHT = 480
NUM_FRAMES = 81
CENTER_FRAME = NUM_FRAMES // 2


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def center_crop_box(width: int, height: int) -> CropBox:
    """Return the largest centered integer crop with the 832:480 ratio."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")
    target_ratio = TARGET_WIDTH / TARGET_HEIGHT
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_height = height
        crop_width = min(width, round(height * target_ratio))
    else:
        crop_width = width
        crop_height = min(height, round(width / target_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return CropBox(left, top, left + crop_width, top + crop_height)


def crop_array(value: np.ndarray, box: CropBox) -> np.ndarray:
    if value.ndim not in (2, 3):
        raise ValueError(f"Expected HW or HWC array, got {value.shape}")
    if value.shape[0] < box.bottom or value.shape[1] < box.right:
        raise ValueError(f"Crop {box} exceeds array shape {value.shape}")
    return np.ascontiguousarray(value[box.top : box.bottom, box.left : box.right])


def image_to_repeated_video(image: Image.Image) -> tuple[np.ndarray, CropBox]:
    """Center-crop without distortion, resize, then repeat into 81 frames."""
    image = image.convert("RGB")
    box = center_crop_box(*image.size)
    cropped = image.crop((box.left, box.top, box.right, box.bottom))
    resized = cropped.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.BICUBIC)
    frame = np.asarray(resized, dtype=np.uint8)
    video = np.repeat(frame[None], NUM_FRAMES, axis=0)
    return video, box


def resize_prediction_to_crop(prediction: np.ndarray, box: CropBox) -> np.ndarray:
    """Resize an HW/HWC model prediction to the original cropped GT size."""
    if prediction.ndim == 2:
        tensor = torch.from_numpy(prediction).float()[None, None]
        resized = F.interpolate(
            tensor, size=(box.height, box.width), mode="bilinear", align_corners=False
        )[0, 0]
        return resized.numpy()
    if prediction.ndim == 3:
        tensor = torch.from_numpy(prediction).permute(2, 0, 1).float()[None]
        resized = F.interpolate(
            tensor, size=(box.height, box.width), mode="bilinear", align_corners=False
        )[0]
        return resized.permute(1, 2, 0).numpy()
    raise ValueError(f"Expected HW or HWC prediction, got {prediction.shape}")


def align_relative_inverse_depth(
    prediction: np.ndarray, ground_truth_depth: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Align a relative inverse-depth prediction to inverse metric GT.

    The fitted scale and shift remove the unknown affine gauge of the model's
    relative inverse-depth representation.  This does not claim that the model
    itself predicts metric depth.
    """
    if prediction.shape != ground_truth_depth.shape or valid.shape != ground_truth_depth.shape:
        raise ValueError("Depth prediction, GT, and mask shapes must match")
    x = prediction[valid].astype(np.float64)
    y = np.reciprocal(ground_truth_depth[valid].astype(np.float64))
    if x.size < 2:
        raise ValueError("Need at least two valid depth pixels")
    design = np.stack((x, np.ones_like(x)), axis=1)
    scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
    aligned_inverse = prediction.astype(np.float64) * scale + shift
    return aligned_inverse, float(scale), float(shift)


def relative_inverse_depth_metrics(
    aligned_inverse: np.ndarray,
    ground_truth_depth: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    """Affine-invariant errors in inverse-depth space.

    Relative RMSE is normalized by the RMS magnitude of GT inverse depth, so
    it is dimensionless and independent of the GT's physical unit.
    """
    gt_inverse = np.reciprocal(ground_truth_depth[valid].astype(np.float64))
    pred_inverse = aligned_inverse[valid].astype(np.float64)
    residual = pred_inverse - gt_inverse
    denominator = np.sqrt(np.mean(gt_inverse**2))
    return {
        "inverse_abs_rel": float(np.mean(np.abs(residual) / gt_inverse)),
        "inverse_relative_rmse": float(np.sqrt(np.mean(residual**2)) / denominator),
        "inverse_positive_fraction": float(np.mean(pred_inverse > 0.0)),
    }


def inverse_depth_to_depth(
    aligned_inverse: np.ndarray, *, min_depth: float = 1e-3, max_depth: float
) -> np.ndarray:
    """Convert aligned inverse depth for bounded conventional depth metrics."""
    if not 0.0 < min_depth < max_depth:
        raise ValueError(f"Invalid depth bounds: {min_depth}..{max_depth}")
    safe_inverse = np.maximum(aligned_inverse.astype(np.float64), 1.0 / max_depth)
    return np.clip(np.reciprocal(safe_inverse), min_depth, max_depth)


def depth_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    gt = ground_truth[valid].astype(np.float64)
    pred = prediction[valid].astype(np.float64)
    pred = np.maximum(pred, 1e-6)
    ratio = np.maximum(gt / pred, pred / gt)
    return {
        "abs_rel": float(np.mean(np.abs(gt - pred) / gt)),
        "rmse": float(np.sqrt(np.mean((gt - pred) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def normalize_normals(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = value.astype(np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    valid = np.isfinite(value).all(axis=-1) & (norm[..., 0] > 1e-8)
    return value / np.maximum(norm, 1e-8), valid


def normal_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    pred, pred_valid = normalize_normals(prediction)
    gt, gt_valid = normalize_normals(ground_truth)
    valid = pred_valid & gt_valid
    if not valid.any():
        raise ValueError("No valid normal pixels")
    cosine = np.clip(np.sum(pred * gt, axis=-1)[valid], -1.0, 1.0)
    error = np.degrees(np.arccos(cosine))
    return {
        "mean_angular_error": float(error.mean()),
        "median_angular_error": float(np.median(error)),
        "within_11_25": float(np.mean(error < 11.25)),
        "within_22_5": float(np.mean(error < 22.5)),
        "within_30": float(np.mean(error < 30.0)),
    }


def matte_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    pred = np.clip(prediction.astype(np.float64), 0.0, 1.0)
    gt = np.clip(ground_truth.astype(np.float64), 0.0, 1.0)
    return {
        "mse": float(np.mean((pred - gt) ** 2)),
        "mad": float(np.mean(np.abs(pred - gt))),
    }
