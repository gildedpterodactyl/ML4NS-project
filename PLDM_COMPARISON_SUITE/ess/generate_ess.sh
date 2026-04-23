#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

N="${N:-50}"
DEVICE="${DEVICE:-cpu}"
ESM_WEIGHT="${ESM_WEIGHT:-0.6}"
REG_WEIGHT="${REG_WEIGHT:-0.4}"
TESS_DELTA="${TESS_DELTA:-6.0}"
USE_ESM2="${USE_ESM2:-false}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/uv-venvs/pldm_comparison_suite}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if [[ "${SKIP_UV_SYNC:-false}" != "true" ]]; then
  uv sync
fi

mkdir -p ess/outputs
uv run --no-sync python generate_sequences.py \
  --mode ess \
  --proldm-root ../PROLDM_OUTLIER \
  --train-csv data/mut_data/GFP-train.csv \
  --ae-ckpt train_logs/GFP/epoch_1000.pt \
  --dataset GFP \
  --n "${N}" \
  --num-chains 4 \
  --burnin 10 \
  --max-steps 1500 \
  --use-esm2 "${USE_ESM2}" \
  --esm-weight "${ESM_WEIGHT}" \
  --reg-weight "${REG_WEIGHT}" \
  --tess-delta "${TESS_DELTA}" \
  --device "${DEVICE}" \
  --out-dir ess/outputs
