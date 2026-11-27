"""Run the resident Wan2.2 driver with chunkwise free-running AR inference."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import infer_wan22_dense4_resident_batch as driver
from train_wan22_ar_tf_ddp import (
    ARTeacherForcingDense4LoRAModule,
    _load_ar_checkpoint,
)
from wan22_cache_integrity import sha256_file
from wan22_dense4_checkpoint import checkpoint_metadata


_build_provenance_metadata = driver.build_provenance_metadata


def _cli_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Missing required AR wrapper argument: {name}") from error


def _audited_build_provenance(*args, code_patch_path, **kwargs):
    result = _build_provenance_metadata(*args, code_patch_path=code_patch_path, **kwargs)
    metadata = checkpoint_metadata(Path(_cli_value("--checkpoint")).resolve())
    actual_patch_hash = sha256_file(Path(code_patch_path).resolve())
    if actual_patch_hash != metadata.get("runtime_source_patch_sha256"):
        raise ValueError("Inference source patch does not match checkpoint metadata")
    result["diffsynth_git_head"] = metadata["diffsynth_git_head"]
    return result


def main() -> None:
    driver.Dense4LoRAModule = ARTeacherForcingDense4LoRAModule
    driver.load_dense4_checkpoint = _load_ar_checkpoint
    driver.build_provenance_metadata = _audited_build_provenance
    driver.main()


if __name__ == "__main__":
    main()
