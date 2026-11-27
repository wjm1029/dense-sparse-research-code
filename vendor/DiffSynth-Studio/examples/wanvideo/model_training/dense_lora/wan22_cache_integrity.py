"""Content-addressed cache contracts shared by Dense4 builders and trainers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


READER_SCHEMA = "wan21_dense4_readers_v2"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_named_files(paths: Iterable[Path], root: Path) -> str:
    """Hash sorted relative names and file bytes, not machine-specific mtimes."""
    digest = hashlib.sha256()
    for path in sorted((Path(item).resolve() for item in paths), key=lambda item: str(item)):
        relative = path.relative_to(root.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def prompt_manifest_hash(prompt_dir: str | Path) -> str:
    root = Path(prompt_dir).resolve()
    paths = list(root.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No prompt JSONL files under {root}")
    return sha256_named_files(paths, root)


def expected_task_paths_hash(task: str, clip_ids: list[str]) -> str:
    return sha256_lines(f"{task}/{clip_id}.pt" for clip_id in clip_ids)


def expected_depth_paths_hash(clip_ids: list[str]) -> str:
    return sha256_lines(f"{clip_id}.pt" for clip_id in clip_ids)


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Cache contract mismatch for {label}: {actual!r} != {expected!r}")
