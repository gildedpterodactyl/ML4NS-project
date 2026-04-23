#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

WITH_STRUCTURE="${WITH_STRUCTURE:-true}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_CHAINS="${NUM_CHAINS:-8}"
DEVICE="${DEVICE:-cpu}"
TESS_DELTA="${TESS_DELTA:-6.0}"
ESM_WEIGHT="${ESM_WEIGHT:-0.6}"
REG_WEIGHT="${REG_WEIGHT:-0.4}"

uv sync

# 1) Baseline PRO-LDM generation (classifier-free guidance label=8)
uv run python baseline_pldm_sample.py \
  --proldm-root ../PROLDM_OUTLIER \
  --checkpoint train_logs/GFP/dropout_tiny_epoch_1000.pt \
  --dataset GFP \
  --num-samples "${NUM_SAMPLES}" \
  --label 8 \
  --identity \
  --device "${DEVICE}" \
  --output-csv generated_seq/raw_baseline_pldm.csv

# 2) ESS generation
uv run python tfg_generation.py \
  --proldm-root ../PROLDM_OUTLIER \
  --checkpoint train_logs/GFP/epoch_1000.pt \
  --input-csv data/mut_data/GFP-train.csv \
  --guidance hybrid \
  --esm2-model facebook/esm2_t6_8M_UR50D \
  --esm-weight "${ESM_WEIGHT}" \
  --reg-weight "${REG_WEIGHT}" \
  --num-samples "${NUM_SAMPLES}" \
  --num-chains "${NUM_CHAINS}" \
  --use-tess false \
  --delta "${TESS_DELTA}" \
  --start-mode random \
  --device "${DEVICE}" \
  --output-csv generated_seq/raw_results_ess.csv \
  --output-json generated_seq/raw_results_ess_meta.json

# 3) TESS generation
uv run python tfg_generation.py \
  --proldm-root ../PROLDM_OUTLIER \
  --checkpoint train_logs/GFP/epoch_1000.pt \
  --input-csv data/mut_data/GFP-train.csv \
  --guidance hybrid \
  --esm2-model facebook/esm2_t6_8M_UR50D \
  --esm-weight "${ESM_WEIGHT}" \
  --reg-weight "${REG_WEIGHT}" \
  --num-samples "${NUM_SAMPLES}" \
  --num-chains "${NUM_CHAINS}" \
  --use-tess true \
  --delta "${TESS_DELTA}" \
  --start-mode random \
  --device "${DEVICE}" \
  --output-csv generated_seq/raw_results_tess.csv \
  --output-json generated_seq/raw_results_tess_meta.json

# 4) Comparison study and standardized output CSVs + plots
uv run python tfg_compare_study.py \
  --proldm-root ../PROLDM_OUTLIER \
  --checkpoint train_logs/GFP/epoch_1000.pt \
  --train-csv data/mut_data/GFP-train.csv \
  --baseline-raw-csv generated_seq/raw_baseline_pldm.csv \
  --ess-raw-csv generated_seq/raw_results_ess.csv \
  --tess-raw-csv generated_seq/raw_results_tess.csv \
  --dataset GFP \
  --umap-train-samples 5000 \
  --esm2-model facebook/esm2_t6_8M_UR50D \
  --device "${DEVICE}" \
  --with-structure "${WITH_STRUCTURE}" \
  --structure-top-k-per-method 100 \
  --results-dir results/comparison_study
