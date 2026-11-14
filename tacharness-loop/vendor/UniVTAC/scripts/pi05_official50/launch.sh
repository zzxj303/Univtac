#!/usr/bin/env bash
set -euo pipefail
TASK="$1"; MODE="$2"; shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU=0; OUTPUT="$ROOT/outputs/pi05_official50/runs/${TASK}_${MODE}"; RESUME=()
while (($#)); do
 case "$1" in
  --gpu) GPU="$2"; shift 2;;
  --output) OUTPUT="$2"; shift 2;;
  --resume) RESUME=(--resume); shift;;
  *) echo "Usage: $0 [--gpu ID] [--output DIRECTORY] [--resume]" >&2; exit 2;;
 esac
done
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=42 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
exec "$ROOT/.venv-pi05/bin/python" -u "$ROOT/scripts/pi05_official50/train.py" --task "$TASK" --mode "$MODE" --output "$OUTPUT" "${RESUME[@]}"
