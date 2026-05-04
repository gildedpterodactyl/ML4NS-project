#!/usr/bin/env bash
# =============================================================================
# run_ress.sh  --  Run RESS (Restart ESS) for one dataset
#
# Usage:
#   bash ress/run_ress.sh                          # GFP, cpu, 1000 seqs
#   N=1000 DEVICE=cuda bash ress/run_ress.sh
#   DATASET=TAPE bash ress/run_ress.sh
#   USE_TRANSPORT=true bash ress/run_ress.sh       # RESS with Transport ESS
# =============================================================================
set -euo pipefail
# Always run from PLDM_COMPARISON_SUITE root
cd "$(dirname "$0")/.."

# -- tuneable defaults --------------------------------------------------------
DATASET="${DATASET:-GFP}"
N="${N:-1000}"
DEVICE="${DEVICE:-cpu}"
USE_TRANSPORT="${USE_TRANSPORT:-false}"

# ESS core
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-1.0}"
TESS_DELTA="${TESS_DELTA:-0.5}"
BURNIN="${BURNIN:-100}"
MAX_STEPS="${MAX_STEPS:-8000}"
NUM_CHAINS="${NUM_CHAINS:-8}"
ALPHA="${ALPHA:-0.0}"

# RESS restart
RESS_ROUNDS="${RESS_ROUNDS:-5}"
RESS_TOP_K="${RESS_TOP_K:-8}"
RESS_MIN_HAMMING="${RESS_MIN_HAMMING:-3}"
RESS_PATIENCE="${RESS_PATIENCE:-50}"
RESS_TEMP_SCALE="${RESS_TEMP_SCALE:-1.05}"

# Transport flow (only used when USE_TRANSPORT=true)
FLOW_BUFFER_SIZE="${FLOW_BUFFER_SIZE:-128}"
FLOW_ADAPT_EVERY="${FLOW_ADAPT_EVERY:-32}"
FLOW_LR="${FLOW_LR:-1e-3}"
FLOW_ADAPT_STEPS="${FLOW_ADAPT_STEPS:-5}"

# -- Dataset registry ---------------------------------------------------------
case "$DATASET" in
  GFP)
    TRAIN_CSV="data/mut_data/GFP-train.csv"
    AE_CKPT="train_logs/GFP/epoch_1000.pt"
    ;;
  TAPE)
    TRAIN_CSV="data/mut_data/TAPE-train.csv"
    AE_CKPT="train_logs/TAPE/epoch_1000.pt"
    ;;
  *)
    echo "Unknown dataset: $DATASET. Add it to the case block in run_ress.sh"
    exit 1
    ;;
esac

METHOD_TAG="ress"
[[ "$USE_TRANSPORT" == "true" ]] && METHOD_TAG="ress_transport"

OUT_DIR="ress/outputs/${DATASET}"
RES_DIR="ress/results/${DATASET}"
mkdir -p "$OUT_DIR" "$RES_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "===== RESS: $DATASET (mode=$METHOD_TAG, N=$N, device=$DEVICE) ====="
log "  rounds=$RESS_ROUNDS  top_k=$RESS_TOP_K  min_hamming=$RESS_MIN_HAMMING"
log "  patience=$RESS_PATIENCE  temp_scale=$RESS_TEMP_SCALE  base_T=$LATENT_TEMPERATURE"
log "  delta=$TESS_DELTA  burnin=$BURNIN  max_steps=$MAX_STEPS"

# -- Generate -----------------------------------------------------------------
log "  generating..."
uv run python ress/generate_ress.py \
  --proldm-root        ../PROLDM_OUTLIER \
  --train-csv          "${TRAIN_CSV}" \
  --ae-ckpt            "${AE_CKPT}" \
  --dataset            "${DATASET}" \
  --n                  "${N}" \
  --num-chains         "${NUM_CHAINS}" \
  --burnin             "${BURNIN}" \
  --max-steps          "${MAX_STEPS}" \
  --latent-temperature "${LATENT_TEMPERATURE}" \
  --tess-delta         "${TESS_DELTA}" \
  --ress-rounds        "${RESS_ROUNDS}" \
  --ress-top-k         "${RESS_TOP_K}" \
  --ress-min-hamming   "${RESS_MIN_HAMMING}" \
  --ress-patience      "${RESS_PATIENCE}" \
  --ress-temp-scale    "${RESS_TEMP_SCALE}" \
  --use-transport      "${USE_TRANSPORT}" \
  --flow-buffer-size   "${FLOW_BUFFER_SIZE}" \
  --flow-adapt-every   "${FLOW_ADAPT_EVERY}" \
  --flow-lr            "${FLOW_LR}" \
  --flow-adapt-steps   "${FLOW_ADAPT_STEPS}" \
  --alpha              "${ALPHA}" \
  --device             "${DEVICE}" \
  --out-dir            "${OUT_DIR}"

log "  generation done."

# -- Evaluate -----------------------------------------------------------------
log "  evaluating..."
uv run python evaluate_method.py \
  --method-name    "${METHOD_TAG}" \
  --input-csv      "${OUT_DIR}/results_${METHOD_TAG}.csv" \
  --results-csv    "${RES_DIR}/results.csv" \
  --proldm-root    ../PROLDM_OUTLIER \
  --train-csv      "${TRAIN_CSV}" \
  --ae-ckpt        "${AE_CKPT}" \
  --dataset        "${DATASET}" \
  --device         "${DEVICE}"

log "  done: ${RES_DIR}/results.csv"

# -- Print scorecard ----------------------------------------------------------
if [[ -f "${RES_DIR}/results.summary.json" ]]; then
  python3 - "${RES_DIR}/results.summary.json" <<'PYEOF'
import sys, json
d = json.load(open(sys.argv[1]))
print("\n=== Scorecard: RESS ===")
fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
for k, v in d.items():
    print(f"  {k:<40}: {fmt(v)}")
PYEOF
fi
