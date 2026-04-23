#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

USE_ESM2="${USE_ESM2:-false}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/uv-venvs/pldm_comparison_suite}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if [[ "${SKIP_UV_SYNC:-false}" != "true" ]]; then
  uv sync
fi

mkdir -p tess/results
uv run --no-sync python evaluate_method.py \
  --method tess \
  --input-csv tess/outputs/results_tess.csv \
  --results-csv tess/results/results.csv \
  --use-esm2 "${USE_ESM2}" \
  --proldm-root ../PROLDM_OUTLIER \
  --dataset GFP
