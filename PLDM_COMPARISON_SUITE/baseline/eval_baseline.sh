#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"
WITH_STRUCTURE="${WITH_STRUCTURE:-false}"
USE_ESM2="${USE_ESM2:-false}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/uv-venvs/pldm_comparison_suite}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if [[ "${SKIP_UV_SYNC:-false}" != "true" ]]; then
  uv sync
fi

mkdir -p baseline/results
uv run --no-sync python evaluate_method.py \
  --method-name baseline_pldm \
  --input-csv baseline/outputs/baseline_pldm.csv \
  --results-csv baseline/results/results.csv \
  --proldm-root ../PROLDM_OUTLIER \
  --train-csv data/mut_data/GFP-train.csv \
  --ae-ckpt train_logs/GFP/epoch_1000.pt \
  --dataset GFP \
  --use-esm2 "${USE_ESM2}" \
  --with-structure "${WITH_STRUCTURE}" \
  --device "${DEVICE}"
