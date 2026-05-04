#!/usr/bin/env bash
# =============================================================================
# run_all_datasets.sh
# Run all 5 modes (baseline, ess, tess, transport_ess, ress) x all datasets,
# then call merge_and_plot.py for combined plots.
#
# All outputs land under a timestamped run directory:
#   results_<YYYYMMDD_HHMMSS>/
#     baseline/
#       outputs/<DATASET>/   ← raw generated CSVs
#       results/<DATASET>/   ← evaluated results.csv + summary.json
#     ess/          (same structure)
#     tess/         (same structure)
#     transport_ess/(same structure)
#     ress/         (same structure)
#     common/<DATASET>/results/  ← merged plots + tables
#     results_summary.tsv        ← cross-method summary table
#
# MODE TAXONOMY (matches generate_sequences.py --mode flag):
#   baseline      : PRO-LDM diffusion sampler (the paper's method, no MCMC)
#   ess           : Elliptical Slice Sampling from top-fitness seed
#   tess          : ESS from WT centroid (different init, same algorithm)
#   transport_ess : ESS with learned RealNVP flow preconditioning (true TESS)
#   ress          : Rejection ESS -- post-hoc hard-threshold filter on ESS samples
#
# Usage:
#   bash run_all_datasets.sh                          # all modes, all datasets, cpu
#   DEVICE=cuda bash run_all_datasets.sh              # gpu
#   DATASETS="GFP" MODES="ess transport_ess" bash run_all_datasets.sh
#   N=500 DEVICE=cuda bash run_all_datasets.sh
#   SKIP_PLOTS=true bash run_all_datasets.sh          # skip merge_and_plot step
#   RUN_DIR=my_run bash run_all_datasets.sh           # override run directory name
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# -- timestamped run directory -----------------------------------------------
# All outputs go under this root so each invocation is self-contained.
# Override by setting RUN_DIR before calling the script.
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ROOT="${RUN_DIR:-results_${TIMESTAMP}}"
mkdir -p "${RUN_ROOT}"
log_file="${RUN_ROOT}/run.log"
echo "Run started at $(date)" | tee "${log_file}"
echo "All outputs will be written to: ${RUN_ROOT}/" | tee -a "${log_file}"

# -- tuneable defaults -------------------------------------------------------
# NOTE: All MCMC defaults are kept in sync with generate_sequences.py parse_args().
#       If you change a default there, update it here too.
N="${N:-1000}"
DEVICE="${DEVICE:-cpu}"
WITH_STRUCTURE="${WITH_STRUCTURE:-false}"
ALPHA="${ALPHA:-0.5}"
# Raised 0.75 -> 1.2 to match generate_sequences.py (wider ellipse, better mixing)
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-1.2}"
ESM_WEIGHT="${ESM_WEIGHT:-0.5}"
REG_WEIGHT="${REG_WEIGHT:-0.5}"
# ESS_DELTA and TESS_DELTA both raised from 0.05/0.14 -> 0.5 to match generate_sequences.py
# (old values covered <1% of typical latent L2 norm, collapsing the bracket on every step)
ESS_DELTA="${ESS_DELTA:-0.5}"
TESS_DELTA="${TESS_DELTA:-0.5}"
# Optional linear anneal target for ball radius (0 = disabled, keep delta fixed)
DELTA_FINAL="${DELTA_FINAL:-1.0}"
OMEGA="${OMEGA:-20.0}"
LOO_MAX_SAMPLES="${LOO_MAX_SAMPLES:-200}"
NUM_CHAINS="${NUM_CHAINS:-8}"
# Raised 20 -> 100 to match generate_sequences.py (EMA normaliser needs warm-up)
BURNIN="${BURNIN:-100}"
# Raised 5000 -> 10000 to match generate_sequences.py (more accepted states per chain)
MAX_STEPS="${MAX_STEPS:-10000}"
FLOW_BUFFER_SIZE="${FLOW_BUFFER_SIZE:-128}"
FLOW_ADAPT_EVERY="${FLOW_ADAPT_EVERY:-32}"
FLOW_LR="${FLOW_LR:-1e-3}"
FLOW_ADAPT_STEPS="${FLOW_ADAPT_STEPS:-5}"
SKIP_UV_SYNC="${SKIP_UV_SYNC:-false}"
SKIP_PLOTS="${SKIP_PLOTS:-false}"

# Modes to run. Override with: MODES="ess transport_ess" bash run_all_datasets.sh
MODES=(${MODES:-baseline ess tess transport_ess ress})

# -- dataset registry --------------------------------------------------------
# Format: "NAME:train_csv:ae_ckpt:baseline_ckpt:wt_pdb"
# All paths relative to ../PROLDM_OUTLIER/
DATASET_REGISTRY=(
  "GFP:data/mut_data/GFP-train.csv:train_logs/GFP/epoch_1000.pt:train_logs/GFP/dropout_tiny_epoch_1000.pt:1GFL"
  "TAPE:data/mut_data/TAPE-train.csv:train_logs/TAPE/epoch_1000.pt:train_logs/TAPE/dropout_tiny_epoch_1000.pt:none"
)

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
  SKIP_UV_SYNC=true
  export SKIP_UV_SYNC
fi

# -- helpers -----------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${log_file}"; }

should_run() {
  local ds="$1"
  [[ ${#FILTER[@]} -eq 0 ]] && return 0
  for f in "${FILTER[@]}"; do [[ "$f" == "$ds" ]] && return 0; done
  return 1
}

mode_requested() {
  local m="$1"
  for x in "${MODES[@]}"; do [[ "$x" == "$m" ]] && return 0; done
  return 1
}

SUMMARY_FILES=()

# -- main loop ---------------------------------------------------------------
for entry in "${DATASET_REGISTRY[@]}"; do
  IFS=: read -r DATASET TRAIN_CSV AE_CKPT BASELINE_CKPT WT_PDB <<< "$entry"

  should_run "$DATASET" || { log "Skipping $DATASET"; continue; }

  if [[ ! -f "../PROLDM_OUTLIER/$TRAIN_CSV" ]]; then
    log "WARNING: $DATASET train CSV missing -- skipping"; continue
  fi
  if [[ ! -f "../PROLDM_OUTLIER/$AE_CKPT" ]]; then
    log "WARNING: $DATASET AE checkpoint missing -- skipping"; continue
  fi

  log "===== DATASET: $DATASET ====="

  STRUCT_FLAG="${WITH_STRUCTURE}"
  if [[ "$WT_PDB" == "none" ]]; then
    STRUCT_FLAG="false"
    log "  No WT PDB -- structure metrics disabled"
  fi

  # ---- BASELINE (PRO-LDM diffusion sampler) --------------------------------
  if mode_requested baseline; then
    log "  -- baseline --"
    OUT_DIR="${RUN_ROOT}/baseline/outputs/${DATASET}"
    RES_DIR="${RUN_ROOT}/baseline/results/${DATASET}"
    mkdir -p "$OUT_DIR" "$RES_DIR"

    if [[ ! -f "../PROLDM_OUTLIER/$BASELINE_CKPT" ]]; then
      log "  WARNING: baseline checkpoint missing at ../PROLDM_OUTLIER/$BASELINE_CKPT -- skipping baseline"
    else
      log "    generating $N sequences..."
      uv run --no-sync python generate_sequences.py \
        --mode          baseline \
        --proldm-root   ../PROLDM_OUTLIER \
        --train-csv     "${TRAIN_CSV}" \
        --ae-ckpt       "${AE_CKPT}" \
        --baseline-ckpt "${BASELINE_CKPT}" \
        --dataset       "${DATASET}" \
        --n             "${N}" \
        --omega         "${OMEGA}" \
        --device        "${DEVICE}" \
        --out-dir       "${OUT_DIR}"

      log "    evaluating..."
      WT_PDB_FLAG=""; [[ "$WT_PDB" != "none" ]] && WT_PDB_FLAG="--wt-pdb ${WT_PDB}"
      uv run --no-sync python evaluate_method.py \
        --method-name     baseline \
        --input-csv       "${OUT_DIR}/baseline_pldm.csv" \
        --results-csv     "${RES_DIR}/results.csv" \
        --proldm-root     ../PROLDM_OUTLIER \
        --train-csv       "${TRAIN_CSV}" \
        --ae-ckpt         "${AE_CKPT}" \
        --dataset         "${DATASET}" \
        --with-structure  "${STRUCT_FLAG}" \
        --loo-max-samples "${LOO_MAX_SAMPLES}" \
        --device          "${DEVICE}" \
        ${WT_PDB_FLAG}

      SUMMARY_FILE="${RES_DIR}/results.summary.json"
      [[ -f "$SUMMARY_FILE" ]] && SUMMARY_FILES+=("${DATASET}__baseline__${SUMMARY_FILE}")
      log "    done: ${RES_DIR}/results.csv"
    fi
  fi

  # ---- ESS / TESS (plain ESS, two different inits) -------------------------
  for METHOD in ess tess; do
    mode_requested "$METHOD" || continue
    log "  -- $METHOD --"

    OUT_DIR="${RUN_ROOT}/${METHOD}/outputs/${DATASET}"
    RES_DIR="${RUN_ROOT}/${METHOD}/results/${DATASET}"
    mkdir -p "$OUT_DIR" "$RES_DIR"

    # ess uses ESS_DELTA; tess uses TESS_DELTA (same algorithm, different center+delta)
    [[ "$METHOD" == "ess" ]] && DELTA="${ESS_DELTA}" || DELTA="${TESS_DELTA}"

    log "    generating $N sequences..."
    uv run --no-sync python generate_sequences.py \
      --mode               "${METHOD}" \
      --proldm-root        ../PROLDM_OUTLIER \
      --train-csv          "${TRAIN_CSV}" \
      --ae-ckpt            "${AE_CKPT}" \
      --baseline-ckpt      "${BASELINE_CKPT}" \
      --dataset            "${DATASET}" \
      --n                  "${N}" \
      --num-chains         "${NUM_CHAINS}" \
      --burnin             "${BURNIN}" \
      --max-steps          "${MAX_STEPS}" \
      --alpha              "${ALPHA}" \
      --latent-temperature "${LATENT_TEMPERATURE}" \
      --esm-weight         "${ESM_WEIGHT}" \
      --reg-weight         "${REG_WEIGHT}" \
      --tess-delta         "${DELTA}" \
      --delta-final        "${DELTA_FINAL}" \
      --use-transport      false \
      --device             "${DEVICE}" \
      --out-dir            "${OUT_DIR}"

    log "    evaluating..."
    WT_PDB_FLAG=""; [[ "$WT_PDB" != "none" ]] && WT_PDB_FLAG="--wt-pdb ${WT_PDB}"
    uv run --no-sync python evaluate_method.py \
      --method-name     "${METHOD}" \
      --input-csv       "${OUT_DIR}/results_${METHOD}.csv" \
      --results-csv     "${RES_DIR}/results.csv" \
      --proldm-root     ../PROLDM_OUTLIER \
      --train-csv       "${TRAIN_CSV}" \
      --ae-ckpt         "${AE_CKPT}" \
      --dataset         "${DATASET}" \
      --with-structure  "${STRUCT_FLAG}" \
      --loo-max-samples "${LOO_MAX_SAMPLES}" \
      --device          "${DEVICE}" \
      ${WT_PDB_FLAG}

    SUMMARY_FILE="${RES_DIR}/results.summary.json"
    [[ -f "$SUMMARY_FILE" ]] && SUMMARY_FILES+=("${DATASET}__${METHOD}__${SUMMARY_FILE}")
    log "    done: ${RES_DIR}/results.csv"
  done

  # ---- TRANSPORT ESS (ESS + learned RealNVP flow preconditioning) ----------
  if mode_requested transport_ess; then
    log "  -- transport_ess --"

    OUT_DIR="${RUN_ROOT}/transport_ess/outputs/${DATASET}"
    RES_DIR="${RUN_ROOT}/transport_ess/results/${DATASET}"
    mkdir -p "$OUT_DIR" "$RES_DIR"

    log "    generating $N sequences (RealNVP flow, buffer=${FLOW_BUFFER_SIZE}, adapt_every=${FLOW_ADAPT_EVERY})..."
    uv run --no-sync python generate_sequences.py \
      --mode               transport_ess \
      --proldm-root        ../PROLDM_OUTLIER \
      --train-csv          "${TRAIN_CSV}" \
      --ae-ckpt            "${AE_CKPT}" \
      --baseline-ckpt      "${BASELINE_CKPT}" \
      --dataset            "${DATASET}" \
      --n                  "${N}" \
      --num-chains         "${NUM_CHAINS}" \
      --burnin             "${BURNIN}" \
      --max-steps          "${MAX_STEPS}" \
      --alpha              "${ALPHA}" \
      --latent-temperature "${LATENT_TEMPERATURE}" \
      --esm-weight         "${ESM_WEIGHT}" \
      --reg-weight         "${REG_WEIGHT}" \
      --tess-delta         "${TESS_DELTA}" \
      --delta-final        "${DELTA_FINAL}" \
      --use-transport      true \
      --flow-buffer-size   "${FLOW_BUFFER_SIZE}" \
      --flow-adapt-every   "${FLOW_ADAPT_EVERY}" \
      --flow-lr            "${FLOW_LR}" \
      --flow-adapt-steps   "${FLOW_ADAPT_STEPS}" \
      --device             "${DEVICE}" \
      --out-dir            "${OUT_DIR}"

    log "    evaluating..."
    WT_PDB_FLAG=""; [[ "$WT_PDB" != "none" ]] && WT_PDB_FLAG="--wt-pdb ${WT_PDB}"
    uv run --no-sync python evaluate_method.py \
      --method-name     transport_ess \
      --input-csv       "${OUT_DIR}/results_transport_ess.csv" \
      --results-csv     "${RES_DIR}/results.csv" \
      --proldm-root     ../PROLDM_OUTLIER \
      --train-csv       "${TRAIN_CSV}" \
      --ae-ckpt         "${AE_CKPT}" \
      --dataset         "${DATASET}" \
      --with-structure  "${STRUCT_FLAG}" \
      --loo-max-samples "${LOO_MAX_SAMPLES}" \
      --device          "${DEVICE}" \
      ${WT_PDB_FLAG}

    SUMMARY_FILE="${RES_DIR}/results.summary.json"
    [[ -f "$SUMMARY_FILE" ]] && SUMMARY_FILES+=("${DATASET}__transport_ess__${SUMMARY_FILE}")
    log "    done: ${RES_DIR}/results.csv"
  fi

  # ---- RESS (Rejection ESS -- ESS pool filtered by a hard score threshold) -
  # RESS runs the same sampler as ESS but uses rejection sampling as a
  # post-hoc filter: only sequences whose regressor score exceeds the
  # top-decile threshold of the training set are kept.  The generate step
  # over-samples by 5x (5*N) so that enough candidates survive the filter.
  # The evaluate step uses method-name "ress" so merge_and_plot can treat it
  # as a separate series.
  if mode_requested ress; then
    log "  -- ress (rejection ESS) --"

    OUT_DIR="${RUN_ROOT}/ress/outputs/${DATASET}"
    RES_DIR="${RUN_ROOT}/ress/results/${DATASET}"
    mkdir -p "$OUT_DIR" "$RES_DIR"

    # Over-sample so the rejection filter still yields N accepted sequences
    RESS_OVERSAMPLE=$((N * 5))
    log "    generating ${RESS_OVERSAMPLE} candidates (will filter down to ~${N})..."
    uv run --no-sync python generate_sequences.py \
      --mode               ess \
      --proldm-root        ../PROLDM_OUTLIER \
      --train-csv          "${TRAIN_CSV}" \
      --ae-ckpt            "${AE_CKPT}" \
      --baseline-ckpt      "${BASELINE_CKPT}" \
      --dataset            "${DATASET}" \
      --n                  "${RESS_OVERSAMPLE}" \
      --num-chains         "${NUM_CHAINS}" \
      --burnin             "${BURNIN}" \
      --max-steps          "${MAX_STEPS}" \
      --alpha              "${ALPHA}" \
      --latent-temperature "${LATENT_TEMPERATURE}" \
      --esm-weight         "${ESM_WEIGHT}" \
      --reg-weight         "${REG_WEIGHT}" \
      --tess-delta         "${ESS_DELTA}" \
      --delta-final        "${DELTA_FINAL}" \
      --use-transport      false \
      --device             "${DEVICE}" \
      --out-dir            "${OUT_DIR}"

    # Rename ess output to ress so evaluate_method.py picks the right file
    if [[ -f "${OUT_DIR}/results_ess.csv" && ! -f "${OUT_DIR}/results_ress.csv" ]]; then
      cp "${OUT_DIR}/results_ess.csv" "${OUT_DIR}/results_ress.csv"
    fi

    log "    evaluating (with rejection filter, method-name=ress)..."
    WT_PDB_FLAG=""; [[ "$WT_PDB" != "none" ]] && WT_PDB_FLAG="--wt-pdb ${WT_PDB}"
    uv run --no-sync python evaluate_method.py \
      --method-name     ress \
      --input-csv       "${OUT_DIR}/results_ress.csv" \
      --results-csv     "${RES_DIR}/results.csv" \
      --proldm-root     ../PROLDM_OUTLIER \
      --train-csv       "${TRAIN_CSV}" \
      --ae-ckpt         "${AE_CKPT}" \
      --dataset         "${DATASET}" \
      --with-structure  "${STRUCT_FLAG}" \
      --loo-max-samples "${LOO_MAX_SAMPLES}" \
      --device          "${DEVICE}" \
      --rejection-filter true \
      --rejection-n      "${N}" \
      ${WT_PDB_FLAG}

    SUMMARY_FILE="${RES_DIR}/results.summary.json"
    [[ -f "$SUMMARY_FILE" ]] && SUMMARY_FILES+=("${DATASET}__ress__${SUMMARY_FILE}")
    log "    done: ${RES_DIR}/results.csv"
  fi

done

# -- final summary table -----------------------------------------------------
if [[ ${#SUMMARY_FILES[@]} -gt 0 ]]; then
  log "===== SUMMARY TABLE ====="
  TSV_OUT="${RUN_ROOT}/results_summary.tsv"

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
    print(f"{ds}\t{method}\tERROR: {e}"); sys.exit(0)
fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else ("nan" if isinstance(v, float) else str(v))
print(ds, method,
    d.get('n_sequences','?'),
    fmt(d.get('mean_internal_regressor_score', float('nan'))),
    fmt(d.get('mean_positional_kl_divergence', float('nan'))),
    fmt(d.get('loo_spearman_rho', float('nan'))),
    fmt(d.get('spearman_rho_vs_truth', float('nan'))),
    fmt(d.get('mean_nearest_train_identity', float('nan'))),
    fmt(d.get('mean_plddt_score', float('nan'))),
    fmt(d.get('ess_per_sec', float('nan'))),
    sep='\t')
PYEOF
  done | tee -a "$TSV_OUT"
  log "Summary saved to: ${TSV_OUT}"
fi

# -- per-dataset merge + plot ------------------------------------------------
if [[ "${SKIP_PLOTS}" != "true" ]]; then
  log "===== PLOTS ====="
  for entry in "${DATASET_REGISTRY[@]}"; do
    IFS=: read -r DATASET TRAIN_CSV AE_CKPT BASELINE_CKPT WT_PDB <<< "$entry"
    should_run "$DATASET" || continue

    BASELINE_RES="${RUN_ROOT}/baseline/results/${DATASET}/results.csv"
    ESS_RES="${RUN_ROOT}/ess/results/${DATASET}/results.csv"
    TESS_RES="${RUN_ROOT}/tess/results/${DATASET}/results.csv"
    TRANSPORT_RES="${RUN_ROOT}/transport_ess/results/${DATASET}/results.csv"
    RESS_RES="${RUN_ROOT}/ress/results/${DATASET}/results.csv"
    PLOTS_DIR="${RUN_ROOT}/common/${DATASET}/results"

    # Only call merge_and_plot if at least the baseline results exist
    if [[ ! -f "$BASELINE_RES" ]]; then
      log "  Skipping plots for $DATASET (no baseline results found)"
      continue
    fi

    mkdir -p "${PLOTS_DIR}"

    WT_PDB_FLAG=""; [[ "$WT_PDB" != "none" ]] && WT_PDB_FLAG="--wt-pdb ${WT_PDB}"
    STRUCT_FLAG="${WITH_STRUCTURE}"; [[ "$WT_PDB" == "none" ]] && STRUCT_FLAG="false"

    log "  Generating plots for $DATASET -> ${PLOTS_DIR}/"
    uv run --no-sync python merge_and_plot.py \
      --proldm-root        ../PROLDM_OUTLIER \
      --train-csv          "${TRAIN_CSV}" \
      --ae-ckpt            "${AE_CKPT}" \
      --dataset            "${DATASET}" \
      --baseline-results   "${BASELINE_RES}" \
      --ess-results        "${ESS_RES}" \
      --tess-results       "${TESS_RES}" \
      --transport-results  "${TRANSPORT_RES}" \
      --ress-results       "${RESS_RES}" \
      --common-dir         "${RUN_ROOT}/common/${DATASET}" \
      --results-dir        "${PLOTS_DIR}" \
      --with-structure     "${STRUCT_FLAG}" \
      --device             "${DEVICE}" \
      ${WT_PDB_FLAG} || log "  WARNING: merge_and_plot failed for $DATASET (non-fatal)"

    log "  Plots saved to: ${PLOTS_DIR}/"
  done
else
  log "SKIP_PLOTS=true -- skipping merge_and_plot step"
fi

log "All done. Run directory: ${RUN_ROOT}/"
