"""Assert exact checkpoint/trainer-state equivalence for uninterrupted vs resumed runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninterrupted", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--step", type=int, required=True)
    return parser.parse_args()


def equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def semantic_loss_rows(path: Path) -> list[tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    columns = (
        "step", "pre_update_total_loss", "pre_update_data_loss", "identity_loss",
        "task_id", "task", "grad_norm", "lora_lr", "router_lr",
    )
    return [tuple(row[column] for column in columns) for row in rows]


def main() -> None:
    args = parse_args()
    roots = [Path(args.uninterrupted).resolve(), Path(args.resumed).resolve()]
    tensors = [load_file(str(root / f"step-{args.step}.safetensors"), device="cpu") for root in roots]
    assert tensors[0].keys() == tensors[1].keys()
    assert all(torch.equal(tensors[0][name], tensors[1][name]) for name in tensors[0])

    states = [torch.load(root / "trainer_state_latest.pt", map_location="cpu", weights_only=False) for root in roots]
    assert equal(states[0], states[1])
    for rank in range(4):
        rng = [
            torch.load(root / f"trainer_rng_rank{rank}_latest.pt", map_location="cpu", weights_only=False)
            for root in roots
        ]
        assert equal(rng[0], rng[1])
    assert semantic_loss_rows(roots[0] / "loss.csv") == semantic_loss_rows(roots[1] / "loss.csv")
    print("Wan2.2 four-GPU uninterrupted/resume equivalence: PASS")


if __name__ == "__main__":
    main()
