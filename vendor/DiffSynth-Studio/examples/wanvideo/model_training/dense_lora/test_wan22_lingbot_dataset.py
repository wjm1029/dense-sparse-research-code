"""Read-only dataset integration smoke test for original + LingBot caches."""

from __future__ import annotations

import argparse

import torch

from train_wan22_dense4_ddp import CachedDense4Dataset, MASK_SHAPE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--target_cache_dir", required=True)
    parser.add_argument("--lingbot_depth_cache_dir", required=True)
    parser.add_argument("--lingbot_normal_cache_dir", required=True)
    parser.add_argument("--lingbot_mask_cache_dir", required=True)
    args = parser.parse_args()
    dataset = CachedDense4Dataset(
        args.depth_cache_dir,
        args.target_cache_dir,
        allow_legacy_v1_cache=True,
        lingbot_depth_cache_dir=args.lingbot_depth_cache_dir,
        lingbot_normal_cache_dir=args.lingbot_normal_cache_dir,
        lingbot_mask_cache_dir=args.lingbot_mask_cache_dir,
    )
    assert len(dataset) == dataset.num_clips * 4 + len(dataset.lingbot_clip_ids) * 2
    assert len(dataset.task_pools[0]) == dataset.num_clips + len(dataset.lingbot_clip_ids)
    assert len(dataset.task_pools[1]) == dataset.num_clips + len(dataset.lingbot_clip_ids)
    assert len(dataset.task_pools[2]) == len(dataset.task_pools[3]) == dataset.num_clips
    base = dataset[0]
    lingbot_depth = dataset[dataset.num_clips * 4]
    lingbot_normal = dataset[dataset.num_clips * 4 + len(dataset.lingbot_clip_ids)]
    assert tuple(base[4].shape) == tuple(lingbot_depth[4].shape) == tuple(lingbot_normal[4].shape) == MASK_SHAPE
    assert torch.equal(base[4], torch.ones_like(base[4]))
    assert torch.equal(lingbot_depth[4], lingbot_normal[4])
    assert 0.0 < float(lingbot_depth[4].mean()) < 1.0
    print(
        f"Wan2.2 LingBot dataset smoke: PASS; total={len(dataset)} "
        f"lingbot={len(dataset.lingbot_clip_ids)} mask_mean={float(lingbot_depth[4].mean()):.6f}"
    )


if __name__ == "__main__":
    main()
