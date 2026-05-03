#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

WITH_STRUCTURE="${WITH_STRUCTURE:-false}"
N="${N:-50}"
DEVICE="${DEVICE:-cpu}"
ESS_DELTA="${ESS_DELTA:-0.05}"
ESS_DELTA_FINAL="${ESS_DELTA_FINAL:-0.04}"
TESS_DELTA="${TESS_DELTA:-0.14}"
TESS_DELTA_FINAL="${TESS_DELTA_FINAL:-0.12}"
ALPHA="${ALPHA:-0.5}"
ESM_WEIGHT="${ESM_WEIGHT:-0.5}"
REG_WEIGHT="${REG_WEIGHT:-0.5}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-0.75}"
USE_ESM2="${USE_ESM2:-true}"
USE_TRANSPORT="${USE_TRANSPORT:-true}"
TRANSPORT_STRENGTH="${TRANSPORT_STRENGTH:-0.5}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/uv-venvs/pldm_comparison_suite}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"

echo "[run_all] starting pipeline (N=${N}, WITH_STRUCTURE=${WITH_STRUCTURE}, DEVICE=${DEVICE}, USE_TRANSPORT=${USE_TRANSPORT}, TRANSPORT_STRENGTH=${TRANSPORT_STRENGTH})"

if [[ "${SKIP_UV_SYNC:-false}" != "true" ]]; then
	echo "[run_all] syncing environment"
	if command -v uv &> /dev/null; then
		uv sync
	else
		echo "[run_all] uv not found, skipping uv sync"
	fi
else
	echo "[run_all] skipping uv sync"
fi

export SKIP_UV_SYNC=true

echo "[run_all] baseline generation/evaluation"
bash baseline/generate_baseline.sh
bash baseline/eval_baseline.sh

echo "[run_all] ESS generation/evaluation"
bash ess/generate_ess.sh
bash ess/eval_ess.sh

echo "[run_all] TESS generation/evaluation"
bash tess/generate_tess.sh
bash tess/eval_tess.sh

echo "[run_all] Transport ESS generation/evaluation"
bash transport/generate_transport.sh
bash transport/eval_transport.sh

echo "[run_all] common plotting"
bash common/run_common_plots.sh

echo "[run_all] done"
