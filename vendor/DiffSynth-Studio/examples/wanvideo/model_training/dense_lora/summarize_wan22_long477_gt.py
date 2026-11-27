"""Aggregate and pair long477 GT JSONL shards from two checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = (
    "mse", "mae", "nmse_target_energy", "mse_first_quarter", "mse_middle_half",
    "mse_last_quarter", "mse_late_over_early", "mse_time_slope",
    "temporal_delta_mse", "temporal_delta_mae", "prediction_delta_l1",
    "pixel_mse", "pixel_mae", "pixel_psnr", "pixel_mse_first_quarter",
    "pixel_mse_last_quarter", "pixel_temporal_delta_mse",
)
NORMAL_METRICS = ("normal_angular_mean_deg", "normal_within_11_25_percent")
HIGHER_IS_BETTER = {"pixel_psnr", "normal_within_11_25_percent"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read(paths: list[str]) -> tuple[dict, dict]:
    headers, rows = [], {}
    for name in paths:
        for line in Path(name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["record_type"] == "header":
                headers.append(row)
            else:
                key = (row["clip_id"], row["task"])
                if key in rows:
                    raise ValueError(f"Duplicate result: {key}")
                rows[key] = row
    if not headers:
        raise ValueError("Missing headers")
    identity_keys = ("label", "checkpoint", "checkpoint_sha256", "checkpoint_step", "selection_seed", "selected_clip_ids", "num_cases")
    for key in identity_keys:
        if any(header[key] != headers[0][key] for header in headers[1:]):
            raise ValueError(f"Shard header mismatch: {key}")
    return headers[0], rows


def aggregate(base: dict, cand: dict, keys: list, metrics=METRICS) -> dict:
    result = {"pairs": len(keys)}
    rng = np.random.default_rng(20260916)
    for metric in metrics:
        b = np.asarray([base[key][metric] for key in keys], dtype=np.float64)
        c = np.asarray([cand[key][metric] for key in keys], dtype=np.float64)
        delta = c - b
        boot = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(10000)])
        result[metric] = {
            "baseline_mean": float(b.mean()), "candidate_mean": float(c.mean()),
            "candidate_minus_baseline": float(delta.mean()),
            "relative_change_percent": float(100.0 * delta.mean() / max(abs(b.mean()), 1e-12)),
            "paired_win_rate_percent": float(
                100.0 * np.mean(c > b if metric in HIGHER_IS_BETTER else c < b)
            ),
            "paired_delta_bootstrap_95ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }
    return result


def main() -> None:
    args = parse_args()
    base_header, base = read(args.baseline)
    cand_header, cand = read(args.candidate)
    if base_header["selected_clip_ids"] != cand_header["selected_clip_ids"]:
        raise ValueError("Checkpoint selections differ")
    if set(base) != set(cand):
        raise ValueError("Paired result keys differ")
    expected = base_header["num_cases"] * 2
    if len(base) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(base)}")
    all_keys = sorted(base)
    report = {
        "schema": "wan22_long477_paired_gt_summary_v1",
        "evaluation_scope": "in_domain_training_pool_not_held_out",
        "baseline": {key: base_header[key] for key in ("label", "checkpoint", "checkpoint_sha256", "checkpoint_step")},
        "candidate": {key: cand_header[key] for key in ("label", "checkpoint", "checkpoint_sha256", "checkpoint_step")},
        "selection_seed": base_header["selection_seed"], "selected_clip_ids": base_header["selected_clip_ids"],
        "combined": aggregate(base, cand, all_keys),
        "by_task": {
            "depth": aggregate(base, cand, [key for key in all_keys if key[1] == "depth"]),
            "normal": aggregate(base, cand, [key for key in all_keys if key[1] == "normal"], METRICS + NORMAL_METRICS),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
