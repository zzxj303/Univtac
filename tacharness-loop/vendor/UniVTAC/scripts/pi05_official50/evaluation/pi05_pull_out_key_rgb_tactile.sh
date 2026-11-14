#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../.." && pwd)"
# Only GPU and output-root may be changed through this fixed entry point.
for arg in "$@"; do
  case "$arg" in
    --case|--case=*) echo "This script fixes its model/task/modality." >&2; exit 2 ;;
  esac
done
exec "$ROOT/.venv-pi05/bin/python" "$HERE/runner.py" --case pi05_pull_out_key_rgb_tactile "$@"
