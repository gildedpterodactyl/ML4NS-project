#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

N="${N:-50}"
DEVICE="${DEVICE:-cpu}"
ESM_WEIGHT="${ESM_WEIGHT:-0.5}"
REG_WEIGHT="${REG_WEIGHT:-0.5}"
ALPHA="${ALPHA:-0.5}"
TESS_DELTA="${TESS_DELTA:-0.14}"
TESS_DELTA_FINAL="${TESS_DELTA_FINAL:-0.12}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-0.75}"
MODEL_TYPE="${MODEL_TYPE:-standard}"
USE_ESM2="${USE_ESM2:-true}"
USE_TRANSPORT="${USE_TRANSPORT:-true}"
TRANSPORT_STRENGTH="${TRANSPORT_STRENGTH:-0.5}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/uv-venvs/pldm_comparison_suite}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if [[ "${SKIP_UV_SYNC:-false}" != "true" ]]; then
  uv sync
fi

mkdir -p transport/outputs

uv run --no-sync python generate_sequences.py \
  --mode transport_ess \
  --proldm-root ../PROLDM_OUTLIER \
  --train-csv data/mut_data/GFP-train.csv \
  --ae-ckpt train_logs/GFP/epoch_1000.pt \
  --model-type "${MODEL_TYPE}" \
  --dataset GFP \
  --n "${N}" \
  --num-chains 8 \
  --burnin 20 \
  --max-steps 5000 \
  --use-esm2 "${USE_ESM2}" \
  --tess-delta "${TESS_DELTA}" \
  --delta-final "${TESS_DELTA_FINAL}" \
  --alpha "${ALPHA}" \
  --latent-temperature "${LATENT_TEMPERATURE}" \
  --esm-weight "${ESM_WEIGHT}" \
  --reg-weight "${REG_WEIGHT}" \
  --transport-strength "${TRANSPORT_STRENGTH}" \
  --device "${DEVICE}" \
  --out-dir transport/outputs
