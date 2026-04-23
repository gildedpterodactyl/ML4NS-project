#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

WITH_STRUCTURE="${WITH_STRUCTURE:-true}"
N="${N:-50}"
DEVICE="${DEVICE:-cpu}"
ESS_DELTA="${ESS_DELTA:-0.05}"
ESS_DELTA_FINAL="${ESS_DELTA_FINAL:-0.04}"
TESS_DELTA="${TESS_DELTA:-0.14}"
TESS_DELTA_FINAL="${TESS_DELTA_FINAL:-0.12}"
TESS_WARP_STRENGTH="${TESS_WARP_STRENGTH:-0.75}"
ALPHA="${ALPHA:-0.5}"
ESM_WEIGHT="${ESM_WEIGHT:-0.5}"
REG_WEIGHT="${REG_WEIGHT:-0.5}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-0.75}"
USE_ESM2="${USE_ESM2:-true}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/uv-venvs/pldm_comparison_suite}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"

export SKIP_UV_SYNC=true

uv sync

bash baseline/generate_baseline.sh
bash baseline/eval_baseline.sh

bash ess/generate_ess.sh
bash ess/eval_ess.sh

bash tess/generate_tess.sh
bash tess/eval_tess.sh

bash common/run_common_plots.sh
