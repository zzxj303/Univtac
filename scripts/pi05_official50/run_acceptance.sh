#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${1:-0}" PYTHONHASHSEED=42 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
PY="$ROOT/.venv-pi05/bin/python"
SCRIPTS="$ROOT/scripts/pi05_official50"
OUT="$ROOT/outputs/pi05_official50"
"$PY" -u "$SCRIPTS/check_data.py" > "$OUT/setup/data_audit.log" 2>&1
"$PY" -u "$SCRIPTS/check_weighting.py" > "$OUT/setup/weighting_audit.log" 2>&1
for MODE in rgb rgb_tactile; do
 "$PY" -u "$SCRIPTS/benchmark.py" --mode "$MODE" > "$OUT/setup/benchmark_${MODE}.log" 2>&1
 "$PY" -u "$SCRIPTS/acceptance.py" --mode "$MODE" --micro 8 > "$OUT/setup/acceptance_${MODE}.log" 2>&1
done
"$PY" -u "$SCRIPTS/seal_acceptance.py" > "$OUT/setup/acceptance_summary.log" 2>&1
