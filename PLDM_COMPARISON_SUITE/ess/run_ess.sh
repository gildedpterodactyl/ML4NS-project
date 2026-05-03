#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

N="${N:-1000}"
DEVICE="${DEVICE:-cpu}"
WITH_STRUCTURE="${WITH_STRUCTURE:-true}"
ESM_WEIGHT="${ESM_WEIGHT:-0.5}"
REG_WEIGHT="${REG_WEIGHT:-0.5}"
ALPHA="${ALPHA:-0.5}"
ESS_DELTA="${ESS_DELTA:-0.05}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-0.75}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/uv-venvs/pldm_comparison_suite}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if [[ "${SKIP_UV_SYNC:-false}" != "true" ]]; then
  uv sync
fi

mkdir -p ess/outputs ess/results

uv run --no-sync python generate_sequences.py \
  --mode ess \
  --proldm-root ../PROLDM_OUTLIER \
  --train-csv data/mut_data/GFP-train.csv \
  --ae-ckpt train_logs/GFP/epoch_1000.pt \
  --dataset GFP \
  --n "${N}" \
  --num-chains 8 \
  --burnin 20 \
  --max-steps 5000 \
  --alpha "${ALPHA}" \
  --latent-temperature "${LATENT_TEMPERATURE}" \
  --esm-weight "${ESM_WEIGHT}" \
  --reg-weight "${REG_WEIGHT}" \
  --tess-delta "${ESS_DELTA}" \
  --device "${DEVICE}" \
  --out-dir ess/outputs

uv run --no-sync python evaluate_method.py \
  --method-name ess \
  --input-csv ess/outputs/results_ess.csv \
  --results-csv ess/results/results.csv \
  --proldm-root ../PROLDM_OUTLIER \
  --train-csv data/mut_data/GFP-train.csv \
  --ae-ckpt train_logs/GFP/epoch_1000.pt \
  --dataset GFP \
  --with-structure "${WITH_STRUCTURE}" \
  --device "${DEVICE}"
