#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
uv sync

mkdir -p generated_seq

for chain in $(seq 0 7); do
  gpu=$((chain % 8))
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python tfg_generation.py \
    --proldm-root ../PROLDM_OUTLIER \
    --checkpoint train_logs/GFP/epoch_1000.pt \
    --input-csv data/mut_data/GFP-train.csv \
    --guidance esm2 \
    --esm2-model facebook/esm2_t6_8M_UR50D \
    --num-samples 125 \
    --num-chains 1 \
    --start-mode random \
    --seed $((42 + chain)) \
    --use-tess true \
    --delta 12.0 \
    --output-csv "generated_seq/tfg_chain_${chain}.csv" \
    --output-json "generated_seq/tfg_chain_${chain}_meta.json" &
done

wait

uv run python - <<'PY'
import glob
import pandas as pd

paths = sorted(glob.glob("generated_seq/tfg_chain_*.csv"))
if not paths:
    raise SystemExit("No chain outputs found.")
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
df.to_csv("generated_seq/tfg_generation_1000.csv", index=False)
print("merged rows:", len(df))
print("saved: generated_seq/tfg_generation_1000.csv")
PY