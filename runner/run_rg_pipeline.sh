#!/usr/bin/env bash
# run_rg_pipeline.sh — Full Rg pipeline for Gradient Ascent, ESS, and TESS
#
# Usage:
#   bash runner/run_rg_pipeline.sh [TARGET_TM] [N_VECTORS] [N_STEPS]
#
# Examples:
#   bash runner/run_rg_pipeline.sh            # defaults: Tm=50, n=100, steps=50
#   bash runner/run_rg_pipeline.sh 70 50 100  # Tm=70, 50 vectors, 100 steps
#
# What it does (per target Tm):
#   1. Run experiment_runner.py  → z_final_{method}.pt for GA, ESS, TESS
#   2. Decode each z tensor → PDB files via protein-verification/decode_latent.py
#   3. Compute Rg for each PDB   → rg_{method}.npy
#   4. Plot results               → runner/plots/

set -euo pipefail

TARGET_TM=${1:-50}
N_VECTORS=${2:-100}
N_STEPS=${3:-50}

CHECKPOINT="regression/outputs/ridge_latent_tm_model.npz"
AE_CHECKPOINT="model/outputs/autoencoder.ckpt"
RESULTS_DIR="runner/results/targets/target_${TARGET_TM}"
OPT_DIR="${RESULTS_DIR}/optimization"
PDB_BASE="${RESULTS_DIR}/pdbs"
PLOTS_DIR="runner/plots"

mkdir -p "${OPT_DIR}" "${PDB_BASE}" "${PLOTS_DIR}"

echo "========================================================"
echo " Rg Pipeline  |  Target Tm = ${TARGET_TM}°C"
echo " n_vectors=${N_VECTORS}  n_steps=${N_STEPS}"
echo "========================================================"

# ── Step 1: Optimization ─────────────────────────────────────────────────
echo ""
echo "[1/4] Running optimization (GA + ESS + TESS)..."
python -m optimization.experiment_runner \
    --checkpoint    "${CHECKPOINT}" \
    --target-tm     "${TARGET_TM}" \
    --n-vectors     "${N_VECTORS}" \
    --n-steps       "${N_STEPS}" \
    --output-dir    "${OPT_DIR}"

# ── Step 2: Decode z → PDB ────────────────────────────────────────────────
for METHOD in gradient_ascent ess tess; do
    Z_FILE="${OPT_DIR}/z_final_${METHOD}.pt"
    PDB_DIR="${PDB_BASE}/${METHOD}"
    mkdir -p "${PDB_DIR}"

    if [ ! -f "${Z_FILE}" ]; then
        echo "  [WARN] ${Z_FILE} not found, skipping decode for ${METHOD}"
        continue
    fi

    echo ""
    echo "[2/4] Decoding ${METHOD} latent vectors → PDB..."
    python protein-verification/decode_latent.py \
        --z-vectors     "${Z_FILE}" \
        --ae-checkpoint "${AE_CHECKPOINT}" \
        --output-dir    "${PDB_DIR}"
done

# ── Step 3: Compute Rg ────────────────────────────────────────────────────
echo ""
echo "[3/4] Computing Radius of Gyration..."
for METHOD in gradient_ascent ess tess; do
    PDB_DIR="${PDB_BASE}/${METHOD}"
    RG_OUT="${OPT_DIR}/rg_${METHOD}.npy"

    if [ ! -d "${PDB_DIR}" ] || [ -z "$(ls -A ${PDB_DIR} 2>/dev/null)" ]; then
        echo "  [WARN] No PDBs found in ${PDB_DIR}, skipping Rg for ${METHOD}"
        continue
    fi

    echo "  Computing Rg for ${METHOD}..."
    python protein-verification/gyr_pred.py \
        --pdb-dir    "${PDB_DIR}" \
        --output-npy "${RG_OUT}"
done

# ── Step 4: Plot ──────────────────────────────────────────────────────────
echo ""
echo "[4/4] Generating comparison plots..."
python runner/plot_rg_methods.py \
    --results-dir "runner/results" \
    --plots-dir   "${PLOTS_DIR}" \
    --targets     "${TARGET_TM}"

echo ""
echo "========================================================"
echo " Done!  Plots saved to ${PLOTS_DIR}"
echo "========================================================"
