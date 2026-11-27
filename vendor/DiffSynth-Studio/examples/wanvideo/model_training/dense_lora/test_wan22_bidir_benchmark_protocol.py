"""Unit tests for the aspect-preserving bidirectional benchmark protocol."""

from __future__ import annotations

import numpy as np
from PIL import Image

from wan22_bidir_benchmark_protocol import (
    CENTER_FRAME,
    NUM_FRAMES,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    align_relative_inverse_depth,
    center_crop_box,
    depth_metrics,
    image_to_repeated_video,
    inverse_depth_to_depth,
    normal_metrics,
    relative_inverse_depth_metrics,
    resize_prediction_to_crop,
)


def main() -> None:
    wide = center_crop_box(1242, 375)
    assert (wide.width, wide.height) == (650, 375)
    assert abs(wide.width / wide.height - TARGET_WIDTH / TARGET_HEIGHT) < 0.002

    tall = center_crop_box(480, 640)
    assert (tall.width, tall.height) == (480, 277)
    assert abs(tall.width / tall.height - TARGET_WIDTH / TARGET_HEIGHT) < 0.004

    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (375, 1242, 3), np.uint8))
    video, box = image_to_repeated_video(image)
    assert video.shape == (NUM_FRAMES, TARGET_HEIGHT, TARGET_WIDTH, 3)
    assert np.array_equal(video[0], video[CENTER_FRAME])
    assert np.array_equal(video[0], video[-1])

    scalar = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)
    assert resize_prediction_to_crop(scalar, box).shape == (box.height, box.width)

    gt = np.linspace(1, 10, 100, dtype=np.float32).reshape(10, 10)
    # The model output is an arbitrary affine encoding of inverse depth.
    pred = (np.reciprocal(gt) - 0.3) / 2
    valid = np.ones_like(gt, dtype=bool)
    aligned_inverse, scale, shift = align_relative_inverse_depth(pred, gt, valid)
    inverse_metrics = relative_inverse_depth_metrics(aligned_inverse, gt, valid)
    metric_prediction = inverse_depth_to_depth(aligned_inverse, max_depth=80.0)
    assert abs(scale - 2.0) < 1e-6
    assert abs(shift - 0.3) < 1e-6
    assert inverse_metrics["inverse_abs_rel"] < 1e-7
    assert inverse_metrics["inverse_relative_rmse"] < 1e-7
    assert depth_metrics(metric_prediction, gt, valid)["delta1"] == 1.0
    assert depth_metrics(metric_prediction, gt, valid)["abs_rel"] < 1e-7

    normal = np.zeros((8, 9, 3), dtype=np.float32)
    normal[..., 2] = 1
    metrics = normal_metrics(normal, normal)
    assert metrics["mean_angular_error"] == 0.0
    assert metrics["within_11_25"] == 1.0

    print(
        {
            "status": "PASS",
            "wide_crop": (wide.left, wide.top, wide.right, wide.bottom),
            "tall_crop": (tall.left, tall.top, tall.right, tall.bottom),
            "video_shape": video.shape,
        }
    )


if __name__ == "__main__":
    main()
