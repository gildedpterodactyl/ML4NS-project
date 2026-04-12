#!/usr/bin/env bash
# run_rg_pipeline.sh — Full Rg pipeline: Gradient Ascent, ESS, TESS
#
# Usage (from repo root):
#   bash runner/run_rg_pipeline.sh [TARGET_TM] [N_VECTORS] [N_STEPS] [TEMPERATURE]
#
# Temperature guidance (intercept ≈ 26.8 Å, coef_norm ≈ 9):
#   T ≈ (intercept - target)^2 / 4
#   target=5  Å → T ≈ 119  (default 120)
#   target=20 Å → T ≈  12
#   target=30 Å → T ≈   3
#   target=50 Å → T ≈ 134  (target > intercept, same formula)
#
# Decoder coordinate rescaling
# ----------------------------
# The ProteinAE decoder outputs 8 CA skeleton points in a normalised
# coordinate frame (range ≈ [-2, 2], raw Rg ≈ 0.4–1.6).  gyr_pred.py
# is called with --decoded so it multiplies by DECODER_SCALE_FACTOR (28.56)
# to convert to physical Angstroms.  The scale factor is derived from the
# 3NIH roundtrip: native Rg = 11.425 Å, decoded raw Rg ≈ 0.400 Å.
#
# To re-derive the scale factor:
#   python protein-verification/calibrate_decoder_scale.py \
#       --reference-pdb runner/data/3NIH.pdb \
#       --ae-checkpoint /path/to/ae_r1_d8_v1.ckpt
# or (without running encode/decode):
#   python protein-verification/calibrate_decoder_scale.py --skip-roundtrip
#
# Environment overrides (optional):
#   DATA_DIR          – where downloaded PDBs live   (default: runner/data)
#   RESULTS_DIR       – where results are written    (default: runner/results)
#   ORACLE_MODEL      – path to ridge Rg .npz        (default: runner/results/optimization/ridge_latent_radius_of_gyration_model.npz)
#   AE_CHECKPOINT     – autoencoder .ckpt path       (auto-detected from model-checkpoints/)
#   AUTOENCODE_ACCELERATOR – gpu | cpu               (default: gpu)
#   GPU_LIST          – comma-separated GPU ids      (auto-detected)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPML_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$GPML_DIR/.." && pwd)"
cd "$GPML_DIR"

# ── Configurable parameters ───────────────────────────────────────────────
TARGET_TM="${1:-50}"
N_VECTORS="${2:-100}"
N_STEPS="${3:-50}"
# Default temperature: (26.8 - TARGET_TM)^2 / 4, floored at 1
_DEFAULT_T="$(awk -v t="$TARGET_TM" 'BEGIN { v=(26.8-t)^2/4; print (v<1)?1:v }' 2>/dev/null || echo 100)"
TEMPERATURE="${4:-${_DEFAULT_T}}"

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
AUTOENCODE_ACCELERATOR="${AUTOENCODE_ACCELERATOR:-gpu}"

OPT_ROOT="$RESULTS_DIR/optimization"
TARGETS_ROOT="$RESULTS_DIR/targets"
PLOTS_DIR="$RESULTS_DIR/plots"
MODEL_DIR="$GPML_DIR/model"

ORACLE_MODEL="${ORACLE_MODEL:-$OPT_ROOT/ridge_latent_radius_of_gyration_model.npz}"

mkdir -p "$DATA_DIR" "$OPT_ROOT" "$TARGETS_ROOT" "$PLOTS_DIR"

# ── Checkpoint resolution ─────────────────────────────────────────────────
if [[ -z "${AE_CHECKPOINT:-}" ]]; then
    for candidate in \
        "$GPML_DIR/model-checkpoints/ae_r1_d8_v1.ckpt" \
        "$ROOT_DIR/model-checkpoints/ae_r1_d8_v1.ckpt" \
        "$ROOT_DIR/ae_r1_d8_v1.ckpt" \
        "$MODEL_DIR/checkpoints/ae_r1_d8_v1.ckpt"; do
        if [[ -f "$candidate" ]]; then
            AE_CHECKPOINT="$candidate"
            break
        fi
    done
fi
if [[ -z "${AE_CHECKPOINT:-}" ]]; then
    echo "ERROR: Cannot find ae_r1_d8_v1.ckpt. Set AE_CHECKPOINT env var." >&2
    exit 1
fi
mkdir -p "$MODEL_DIR/checkpoints"
ln -sf "$AE_CHECKPOINT" "$MODEL_DIR/checkpoints/ae_r1_d8_v1.ckpt"
echo "Using checkpoint: $AE_CHECKPOINT"

# ── Auto-calibrate decoder scale if not yet done ──────────────────────────
SCALE_FILE="$GPML_DIR/protein-verification/decoder_scale.txt"
if [[ ! -f "$SCALE_FILE" ]]; then
    echo "[calibrate] decoder_scale.txt not found — writing empirical scale (28.56)..."
    python "$GPML_DIR/protein-verification/calibrate_decoder_scale.py" --skip-roundtrip
fi
echo "Decoder scale factor: $(cat "$SCALE_FILE")"

# ── GPU resolution ────────────────────────────────────────────────────────
if [[ "$AUTOENCODE_ACCELERATOR" == "cpu" ]]; then
    GPU_IDS=("cpu")
else
    GPU_LIST_RAW="${GPU_LIST:-}"
    if [[ -z "$GPU_LIST_RAW" ]]; then
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            GPU_LIST_RAW="$CUDA_VISIBLE_DEVICES"
        elif command -v nvidia-smi >/dev/null 2>&1; then
            GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
            [[ "$GPU_COUNT" -gt 0 ]] || { echo "ERROR: nvidia-smi found no GPUs" >&2; exit 1; }
            GPU_LIST_RAW="$(seq -s, 0 $((GPU_COUNT - 1)))"
        else
            echo "WARN: No GPU detected, falling back to CPU." >&2
            GPU_IDS=("cpu")
            AUTOENCODE_ACCELERATOR="cpu"
        fi
    fi
    if [[ "${AUTOENCODE_ACCELERATOR}" != "cpu" ]]; then
        IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST_RAW"
    fi
fi
GPU0="${GPU_IDS[0]}"
echo "Decoder device: $GPU0"

# ── Helper: run autoencoder decode/encode ─────────────────────────────────
run_autoencode() {
    local input_path="$1" output_dir="$2" mode="$3" gpu_id="$4"
    (
        cd "$MODEL_DIR"
        if [[ "$gpu_id" == "cpu" ]]; then
            PYTHONPATH="$MODEL_DIR:${PYTHONPATH:-}" \
            AUTOENCODE_ACCELERATOR="cpu" \
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
            python proteinfoundation/autoencode.py \
                --input_pdb "$input_path" \
                --output_dir "$output_dir" \
                --mode "$mode" \
                --config_path "$MODEL_DIR/configs" \
                --config_name inference_proteinae
        else
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            PYTHONPATH="$MODEL_DIR:${PYTHONPATH:-}" \
            AUTOENCODE_ACCELERATOR="$AUTOENCODE_ACCELERATOR" \
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
            python proteinfoundation/autoencode.py \
                --input_pdb "$input_path" \
                --output_dir "$output_dir" \
                --mode "$mode" \
                --config_path "$MODEL_DIR/configs" \
                --config_name inference_proteinae
        fi
    )
}

# ── Helper: split a [N, d] .pt → N per-row latent dirs ───────────────────
split_latent_tensor() {
    local latent_pt="$1" output_root="$2"
    python - "$latent_pt" "$output_root" <<'PY'
import sys, torch
from pathlib import Path
latent_pt   = Path(sys.argv[1])
output_root = Path(sys.argv[2])
output_root.mkdir(parents=True, exist_ok=True)
z = torch.load(latent_pt, map_location="cpu")
if z.ndim == 1:
    z = z.unsqueeze(0)
for i, row in enumerate(z, start=1):
    d = output_root / f"{i:04d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(row.detach().cpu(), d / "latent_repr.pt")
print(f"Split {len(z)} latent vectors → {output_root}")
PY
}

# ── Helper: decode a .pt tensor → flat dir of sample.pdb files ───────────
decode_latent_tensor() {
    local latent_pt="$1" input_root="$2" output_root="$3" gpu_id="$4"
    rm -rf "$input_root" "$output_root"
    mkdir -p "$input_root" "$output_root"
    split_latent_tensor "$latent_pt" "$input_root"
    for latent_dir in "$input_root"/*/; do
        [[ -d "$latent_dir" ]] || continue
        run_autoencode "$latent_dir" "$output_root" decode "$gpu_id"
    done
}

# ── Helper: copy sample.pdb files to a flat dir ───────────────────────────
flatten_generated_pdbs() {
    local source_root="$1" flat_root="$2"
    rm -rf "$flat_root"; mkdir -p "$flat_root"
    find "$source_root" -type f -name "sample.pdb" | while read -r pdb; do
        parent="$(basename "$(dirname "$pdb")")"
        cp "$pdb" "$flat_root/${parent}.pdb"
    done
}

# ── Helper: run gyr_pred.py with decoder rescaling ────────────────────────
# All decoded PDBs are passed with --decoded so the raw Cα Rg (in normalised
# [-2,2] units) is multiplied by DECODER_SCALE_FACTOR → physical Angstroms.
compute_rg() {
    local pdb_dir="$1" label="$2" out_csv="$3" out_npy="$4"
    python "$GPML_DIR/protein-verification/gyr_pred.py" \
        --input_dir   "$pdb_dir" \
        --label       "$label" \
        --output_csv  "$out_csv" \
        --decoded
    python - "$out_csv" "$out_npy" <<'PY'
import sys, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1])
np.save(sys.argv[2], df["rg"].values)
print(f"Saved {len(df)} Rg values → {sys.argv[2]}")
PY
}

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================================"
echo "  Rg Pipeline  |  Target Tm = ${TARGET_TM}°C"
echo "  n_vectors=${N_VECTORS}  n_steps=${N_STEPS}  temperature=${TEMPERATURE}"
echo "========================================================"

TAG="${TARGET_TM}"
T_DIR="$TARGETS_ROOT/target_${TAG}"
OPT_DIR="$T_DIR/optimization"
PDB_BASE="$T_DIR/pdbs"
VAL_DIR="$T_DIR/validation"
mkdir -p "$OPT_DIR" "$PDB_BASE" "$VAL_DIR"

# ── Step 1: Optimization ──────────────────────────────────────────────────
echo ""
echo "[1/4] Running optimization (GA + ESS + TESS)..."
PYTHONPATH="$GPML_DIR:${PYTHONPATH:-}" python -m optimization.experiment_runner \
    --checkpoint  "$ORACLE_MODEL" \
    --target-tm   "$TARGET_TM" \
    --n-vectors   "$N_VECTORS" \
    --n-steps     "$N_STEPS" \
    --temperature "$TEMPERATURE" \
    --output-dir  "$OPT_DIR"

# ── Step 2: Decode z → PDB ────────────────────────────────────────────────
echo ""
echo "[2/4] Decoding latent vectors → PDB structures..."

declare -A Z_STEMS=(
    [gradient_ascent]="z_final_gradient_ascent"
    [ess]="z_final_ess"
    [tess]="z_final_tess"
)

for METHOD in gradient_ascent ess tess; do
    Z_FILE="$OPT_DIR/${Z_STEMS[$METHOD]}.pt"
    if [[ ! -f "$Z_FILE" ]]; then
        echo "  [WARN] $Z_FILE not found — skipping decode for $METHOD"
        continue
    fi

    INPUT_ROOT="$DATA_DIR/latent_inputs/target_${TAG}/${METHOD}"
    RAW_DIR="$T_DIR/generated_${METHOD}_raw"
    FLAT_DIR="$PDB_BASE/${METHOD}"

    echo "  Decoding $METHOD..."
    decode_latent_tensor "$Z_FILE" "$INPUT_ROOT" "$RAW_DIR" "$GPU0"
    flatten_generated_pdbs "$RAW_DIR" "$FLAT_DIR"
    echo "  → PDBs in $FLAT_DIR"
done

# ── Step 3: Compute Rg (with decoder rescaling) ───────────────────────────
echo ""
echo "[3/4] Computing Radius of Gyration (decoded Cα ×$(cat $SCALE_FILE) → Å)..."

for METHOD in gradient_ascent ess tess; do
    FLAT_DIR="$PDB_BASE/${METHOD}"
    if [[ ! -d "$FLAT_DIR" ]] || [[ -z "$(ls -A "$FLAT_DIR" 2>/dev/null)" ]]; then
        echo "  [WARN] No PDBs in $FLAT_DIR — skipping Rg for $METHOD"
        continue
    fi
    echo "  Computing Rg for $METHOD..."
    compute_rg \
        "$FLAT_DIR" \
        "${METHOD}_t${TAG}" \
        "$VAL_DIR/rg_${METHOD}.csv" \
        "$OPT_DIR/rg_${METHOD}.npy"
done

# ── Step 4: Plot ──────────────────────────────────────────────────────────
echo ""
echo "[4/4] Generating comparison plots..."
python "$SCRIPT_DIR/plot_rg_methods.py" \
    --results-dir "$RESULTS_DIR" \
    --plots-dir   "$PLOTS_DIR" \
    --targets     "$TARGET_TM"

echo ""
echo "========================================================"
echo "  Done!  Plots → $PLOTS_DIR"
echo "========================================================"
