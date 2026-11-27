"""Run a training script with the CUDA 13 runtime and pure-Python fallbacks.

The cog environment supplies the preferred CUDA 13 / NCCL runtime. Its own
packages take precedence; the Wan2.2 environment path is appended only to
resolve missing pure-Python project dependencies such as modelscope.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: cog_runtime_bootstrap.py SCRIPT [ARGS ...]")
    script = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(Path.cwd()))
    sys.path.append("/pfs/mayuema/dense/wan/envs/wan22/lib/python3.11/site-packages")
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
