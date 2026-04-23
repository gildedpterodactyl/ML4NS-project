#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

WITH_STRUCTURE="${WITH_STRUCTURE:-false}"
N="${N:-1000}"
DEVICE="${DEVICE:-cpu}"
TESS_DELTA="${TESS_DELTA:-6.0}"
ESM_WEIGHT="${ESM_WEIGHT:-0.6}"
REG_WEIGHT="${REG_WEIGHT:-0.4}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/uv-venvs/pldm_comparison_suite}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"

uv sync
export SKIP_UV_SYNC=true

bash baseline/run_baseline.sh
bash ess/run_ess.sh
bash tess/run_tess.sh
bash common/run_common_plots.sh
