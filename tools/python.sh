#!/usr/bin/env bash
# Convenient launcher on the existing server; does not install/change packages.
set -euo pipefail
TASK_PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PROJECT_DIR="${DENSE_PROJECT_ROOT:-$(dirname -- "$(dirname -- "$TASK_PACKAGE_DIR")")}"
if [[ -f "$TASK_PROJECT_DIR/tools/wan22_python.sh" ]]; then
  exec bash "$TASK_PROJECT_DIR/tools/wan22_python.sh" "$@"
fi
if [[ -n "${DENSE_PYTHON:-}" && -x "$DENSE_PYTHON" ]]; then
  exec "$DENSE_PYTHON" "$@"
fi
echo "Set DENSE_PROJECT_ROOT to the video_dense project, or DENSE_PYTHON to an existing Python 3.10 environment." >&2
exit 2
