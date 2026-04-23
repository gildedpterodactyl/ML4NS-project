#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

N="${N:-50}"
DEVICE="${DEVICE:-cpu}"
BASELINE_MODEL_TYPE="${BASELINE_MODEL_TYPE:-tiny}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/uv-venvs/pldm_comparison_suite}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if [[ "${SKIP_UV_SYNC:-false}" != "true" ]]; then
  uv sync
fi

mkdir -p baseline/outputs
uv run --no-sync python generate_sequences.py \
  --mode baseline \
  --proldm-root ../PROLDM_OUTLIER \
  --train-csv data/mut_data/GFP-train.csv \
  --baseline-ckpt train_logs/GFP/dropout_tiny_epoch_1000.pt \
  --baseline-model-type "${BASELINE_MODEL_TYPE}" \
  --dataset GFP \
  --n "${N}" \
  --omega 20.0 \
  --device "${DEVICE}" \
  --out-dir baseline/outputs
