#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

uv sync

# local smoke test
uv run python tfg_generation.py \
  --proldm-root ../PROLDM_OUTLIER \
  --checkpoint train_logs/GFP/epoch_1000.pt \
  --input-csv data/mut_data/GFP-train.csv \
  --guidance regressor \
  --num-samples 16 \
  --num-chains 2 \
  --burnin 2 \
  --max-steps-per-chain 80 \
  --output-csv generated_seq/tfg_smoke.csv \
  --output-json generated_seq/tfg_smoke_meta.json

# full local generation (single-process, 8 independent chains)
# uv run python tfg_generation.py \
#   --proldm-root ../PROLDM_OUTLIER \
#   --checkpoint train_logs/GFP/epoch_1000.pt \
#   --input-csv data/mut_data/GFP-train.csv \
#   --guidance esm2 \
#   --esm2-model facebook/esm2_t6_8M_UR50D \
#   --num-samples 1000 \
#   --num-chains 8 \
#   --use-tess true \
#   --delta 12.0 \
#   --output-csv generated_seq/tfg_generation_1000.csv \
#   --output-json generated_seq/tfg_generation_1000_meta.json