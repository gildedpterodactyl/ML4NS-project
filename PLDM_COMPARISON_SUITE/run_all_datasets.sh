#!/usr/bin/env bash
# =============================================================================
# run_all_datasets.sh
# Run the full ESS + TESS pipeline for every dataset found in data/mut_data/.
#
# Usage:
#   bash run_all_datasets.sh                     # all datasets, cpu
#   DEVICE=cuda bash run_all_datasets.sh         # gpu
#   DATASETS="GFP" bash run_all_datasets.sh      # single dataset
#   N=500 DEVICE=cuda bash run_all_datasets.sh   # 500 samples, gpu
#
# Outputs land in:
#   ess/outputs/<DATASET>/   ess/results/<DATASET>/
#   tess/outputs/<DATASET>/  tess/results/<DATASET>/
#
# A final summary table is printed and saved to results_summary.tsv.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# -- tuneable defaults (override via env) ------------------------------------
N="${N:-1000}"
DEVICE="${DEVICE:-cpu}"
WITH_STRUCTURE="${WITH_STRUCTURE:-false}"
ALPHA="${ALPHA:-0.5}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-0.75}"
ESM_WEIGHT="${ESM_WEIGHT:-0.5}"
REG_WEIGHT="${REG_WEIGHT:-0.5}"
ESS_DELTA="${ESS_DELTA:-0.05}"
TESS_DELTA="${TESS_DELTA:-0.14}"
LOO_MAX_SAMPLES="${LOO_MAX_SAMPLES:-200}"
NUM_CHAINS="${NUM_CHAINS:-8}"
BURNIN="${BURNIN:-20}"
MAX_STEPS="${MAX_STEPS:-5000}"
SKIP_UV_SYNC="${SKIP_UV_SYNC:-false}"

# -- dataset registry --------------------------------------------------------
# Format: "DATASET_NAME:train_csv:ae_ckpt:wt_pdb"
# train_csv and ae_ckpt are relative to ../PROLDM_OUTLIER/
# Add new rows here to run on additional datasets.
DATASET_REGISTRY=(
  "GFP:data/mut_data/GFP-train.csv:train_logs/GFP/epoch_1000.pt:1GFL"
  "TAPE:data/mut_data/TAPE-train.csv:train_logs/TAPE/epoch_1000.pt:none"
)

# Allow caller to restrict datasets: DATASETS="GFP TAPE" bash run_all_datasets.sh
if [[ -n "${DATASETS:-}" ]]; then
  FILTER=(${DATASETS})
else
  FILTER=()
fi

# -- uv setup (once) ---------------------------------------------------------
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER/uv-venvs/pldm_comparison_suite}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

if [[ "${SKIP_UV_SYNC}" != "true" ]]; then
  uv sync
  SKIP_UV_SYNC=true   # only sync once across the loop
  export SKIP_UV_SYNC
fi

# -- helpers -----------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

should_run() {
  local ds="$1"
  if [[ ${#FILTER[@]} -eq 0 ]]; then return 0; fi
  for f in "${FILTER[@]}"; do
    [[ "$f" == "$ds" ]] && return 0
  done
  return 1
}

SUMMARY_FILES=()

# -- main loop ---------------------------------------------------------------
for entry in "${DATASET_REGISTRY[@]}"; do
  IFS=: read -r DATASET TRAIN_CSV AE_CKPT WT_PDB <<< "$entry"

  should_run "$DATASET" || { log "Skipping $DATASET (not in DATASETS filter)"; continue; }

  if [[ ! -f "../PROLDM_OUTLIER/$TRAIN_CSV" ]]; then
    log "WARNING: $DATASET train CSV not found at ../PROLDM_OUTLIER/$TRAIN_CSV -- skipping"
    continue
  fi
  if [[ ! -f "../PROLDM_OUTLIER/$AE_CKPT" ]]; then
    log "WARNING: $DATASET checkpoint not found at ../PROLDM_OUTLIER/$AE_CKPT -- skipping"
    continue
  fi

  log "===== DATASET: $DATASET ====="

  STRUCT_FLAG="${WITH_STRUCTURE}"
  if [[ "$WT_PDB" == "none" ]]; then
    STRUCT_FLAG="false"
    log "  No WT PDB for $DATASET -- structure metrics disabled"
  fi

  for METHOD in ess tess; do
    log "  -- $METHOD --"

    OUT_DIR="${METHOD}/outputs/${DATASET}"
    RES_DIR="${METHOD}/results/${DATASET}"
    mkdir -p "$OUT_DIR" "$RES_DIR"

    if [[ "$METHOD" == "ess" ]]; then
      DELTA_FLAG="--tess-delta ${ESS_DELTA}"
    else
      DELTA_FLAG="--tess-delta ${TESS_DELTA}"
    fi

    # -- 1. generate ---------------------------------------------------------
    log "    generating $N sequences..."
    uv run --no-sync python generate_sequences.py \
      --mode          "${METHOD}" \
      --proldm-root   ../PROLDM_OUTLIER \
      --train-csv     "${TRAIN_CSV}" \
      --ae-ckpt       "${AE_CKPT}" \
      --dataset       "${DATASET}" \
      --n             "${N}" \
      --num-chains    "${NUM_CHAINS}" \
      --burnin        "${BURNIN}" \
      --max-steps     "${MAX_STEPS}" \
      --alpha         "${ALPHA}" \
      --latent-temperature "${LATENT_TEMPERATURE}" \
      --esm-weight    "${ESM_WEIGHT}" \
      --reg-weight    "${REG_WEIGHT}" \
      ${DELTA_FLAG} \
      --device        "${DEVICE}" \
      --out-dir       "${OUT_DIR}"

    INPUT_CSV="${OUT_DIR}/results_${METHOD}.csv"
    RESULTS_CSV="${RES_DIR}/results.csv"

    # -- 2. evaluate ---------------------------------------------------------
    log "    evaluating..."
    WT_PDB_FLAG=""
    if [[ "$WT_PDB" != "none" ]]; then
      WT_PDB_FLAG="--wt-pdb ${WT_PDB}"
    fi

    uv run --no-sync python evaluate_method.py \
      --method-name     "${METHOD}" \
      --input-csv       "${INPUT_CSV}" \
      --results-csv     "${RESULTS_CSV}" \
      --proldm-root     ../PROLDM_OUTLIER \
      --train-csv       "${TRAIN_CSV}" \
      --ae-ckpt         "${AE_CKPT}" \
      --dataset         "${DATASET}" \
      --with-structure  "${STRUCT_FLAG}" \
      --loo-max-samples "${LOO_MAX_SAMPLES}" \
      --device          "${DEVICE}" \
      ${WT_PDB_FLAG}

    SUMMARY_FILE="${RES_DIR}/results.summary.json"
    if [[ -f "$SUMMARY_FILE" ]]; then
      SUMMARY_FILES+=("${DATASET}__${METHOD}__${SUMMARY_FILE}")
    fi

    log "    done: ${RESULTS_CSV}"
  done
done

# -- final summary table -----------------------------------------------------
if [[ ${#SUMMARY_FILES[@]} -gt 0 ]]; then
  log "===== SUMMARY TABLE ====="
  TSV_OUT="results_summary.tsv"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "dataset" "method" "n_sequences" \
    "mean_regressor_score" "mean_kl_div" \
    "loo_spearman_rho" "spearman_vs_truth" \
    "mean_nearest_train_identity" "mean_plddt" "ess_per_sec" \
    | tee "$TSV_OUT"

  for entry in "${SUMMARY_FILES[@]}"; do
    DS="${entry%%__*}"
    rest="${entry#*__}"
    MTH="${rest%%__*}"
    JSON_PATH="${rest#*__}"

    python3 - "$DS" "$MTH" "$JSON_PATH" <<'PYEOF'
import sys, json, math
ds, method, path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"{ds}\t{method}\tERROR: {e}")
    sys.exit(0)
fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else ("nan" if isinstance(v, float) else str(v))
print(
    ds, method,
    d.get('n_sequences', '?'),
    fmt(d.get('mean_internal_regressor_score', float('nan'))),
    fmt(d.get('mean_positional_kl_divergence', float('nan'))),
    fmt(d.get('loo_spearman_rho', float('nan'))),
    fmt(d.get('spearman_rho_vs_truth', float('nan'))),
    fmt(d.get('mean_nearest_train_identity', float('nan'))),
    fmt(d.get('mean_plddt_score', float('nan'))),
    fmt(d.get('ess_per_sec', float('nan'))),
    sep='\t'
)
PYEOF
  done | tee -a "$TSV_OUT"

  log "Summary saved to: $TSV_OUT"
fi

log "All done."
