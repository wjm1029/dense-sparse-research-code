#!/usr/bin/env python3
"""Fully deserialize and validate LingBot Wan2.2 depth/normal latent caches."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch


SHAPE = (48, 21, 30, 52)
DEPTH_SCHEMA = "wan22_depth_latent_v2"
NORMAL_SCHEMA = "wan22_lingbot_normal_latent_v1"


def valid_latent(value: object) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == SHAPE
        and value.dtype == torch.bfloat16
        and torch.isfinite(value).all().item()
    )


def validate_pair(job: tuple[str, str, str, str, str]) -> dict:
    clip_id, depth_path, normal_path, depth_contract, normal_contract = job
    errors = []
    try:
        depth = torch.load(depth_path, map_location="cpu", weights_only=False)
        if depth.get("schema") != DEPTH_SCHEMA:
            errors.append("depth schema")
        if depth.get("cache_contract_hash") != depth_contract:
            errors.append("depth contract")
        if depth.get("clip_id") != clip_id or depth.get("prompt") != "output: depth":
            errors.append("depth identity/prompt")
        if not valid_latent(depth.get("source_latent")):
            errors.append("source latent")
        if not valid_latent(depth.get("target_latent")):
            errors.append("depth target latent")
    except Exception as error:
        errors.append(f"depth load: {error!r}")
    try:
        normal = torch.load(normal_path, map_location="cpu", weights_only=False)
        if normal.get("schema") != NORMAL_SCHEMA:
            errors.append("normal schema")
        if normal.get("cache_contract_hash") != normal_contract:
            errors.append("normal contract")
        if (
            normal.get("clip_id") != clip_id
            or normal.get("task") != "normal"
            or normal.get("task_id") != 1
            or normal.get("prompt") != "output: surface normal"
        ):
            errors.append("normal identity/task/prompt")
        if not valid_latent(normal.get("target_latent")):
            errors.append("normal target latent")
    except Exception as error:
        errors.append(f"normal load: {error!r}")
    return {"clip_id": clip_id, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_cache_dir", required=True)
    parser.add_argument("--normal_cache_dir", required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    depth_root, normal_root = Path(args.depth_cache_dir).resolve(), Path(args.normal_cache_dir).resolve()
    depth_manifest_path, normal_manifest_path = depth_root / "manifest.json", normal_root / "manifest.json"
    depth_manifest = json.loads(depth_manifest_path.read_text(encoding="utf-8"))
    normal_manifest = json.loads(normal_manifest_path.read_text(encoding="utf-8"))
    depth_complete = json.loads((depth_root / "CACHE_COMPLETE").read_text(encoding="utf-8"))
    normal_complete = json.loads((normal_root / "CACHE_COMPLETE").read_text(encoding="utf-8"))
    index = json.loads((depth_root / "dataset_index.json").read_text(encoding="utf-8"))
    count = int(depth_manifest["num_samples"])
    clip_ids = [row["id"] for row in index["samples"][:count]]
    contract_errors = []
    if depth_manifest.get("schema") != DEPTH_SCHEMA or depth_complete.get("schema") != DEPTH_SCHEMA:
        contract_errors.append("depth schema")
    if normal_manifest.get("schema") != NORMAL_SCHEMA or normal_complete.get("schema") != NORMAL_SCHEMA:
        contract_errors.append("normal schema")
    if normal_manifest.get("source_cache_contract_hash") != depth_manifest.get("cache_contract_hash"):
        contract_errors.append("normal does not bind depth contract")
    if normal_manifest.get("num_samples") != count:
        contract_errors.append("sample count mismatch")
    if depth_complete.get("num_samples") != count or normal_complete.get("num_samples") != count:
        contract_errors.append("complete marker count")
    expected = {f"{clip_id}.pt" for clip_id in clip_ids}
    actual_depth = {path.relative_to(depth_root).as_posix() for path in depth_root.glob("*/*.pt")}
    actual_normal = {path.relative_to(normal_root).as_posix() for path in normal_root.glob("*/*.pt")}
    if actual_depth != expected:
        contract_errors.append("depth path set")
    if actual_normal != expected:
        contract_errors.append("normal path set")

    jobs = []
    for clip_id in clip_ids:
        scene, clip = clip_id.split("/", 1)
        jobs.append((
            clip_id,
            str(depth_root / scene / f"{clip}.pt"),
            str(normal_root / scene / f"{clip}.pt"),
            str(depth_manifest["cache_contract_hash"]),
            str(normal_manifest["cache_contract_hash"]),
        ))
    failures = []
    checked = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(validate_pair, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            checked += 1
            if result["errors"]:
                failures.append(result)
            if checked % 100 == 0 or checked == len(jobs):
                print(f"checked={checked}/{len(jobs)} failures={len(failures)}", flush=True)
    report = {
        "schema": "wan22-lingbot-latent-cache-validation-v1",
        "status": "passed" if not contract_errors and not failures else "failed",
        "clips": len(jobs),
        "deserialized_payloads": len(jobs) * 2,
        "validated_latents": len(jobs) * 3,
        "latent_shape": list(SHAPE),
        "latent_dtype": "bfloat16",
        "contract_errors": contract_errors,
        "payload_failures": failures,
    }
    (normal_root / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
