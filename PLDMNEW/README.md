# PLDMNEW: TFG ESS/TESS Generation

This directory now contains a local generation pipeline that:

1. Loads the JTAE autoencoder source space from PROLDM checkpoint `train_logs/GFP/epoch_1000.pt`.
2. Defines likelihood guidance using either:
   - ESM-2 (`--guidance esm2`) from decoded sequences, or
   - JTAE regressor (`--guidance regressor`) for quick smoke testing.
3. Runs ESS or TESS (`--use-tess true --delta <radius>`) in latent space.
4. Writes sampled sequences and metadata to `generated_seq/`.

## Paths used from PROLDM

- Checkpoint: `../PROLDM_OUTLIER/train_logs/GFP/epoch_1000.pt`
- Input data: `../PROLDM_OUTLIER/data/mut_data/GFP-train.csv`

## Quick local run with uv

```bash
cd ML4NS-project/PLDMNEW
bash run_tfg_generation_uv.sh
```

This runs a small regressor-guided smoke test and writes:

- `generated_seq/tfg_smoke.csv`
- `generated_seq/tfg_smoke_meta.json`

## Full 1,000 sample run

```bash
cd ML4NS-project/PLDMNEW
uv sync
uv run python tfg_generation.py \
  --proldm-root ../PROLDM_OUTLIER \
  --checkpoint train_logs/GFP/epoch_1000.pt \
  --input-csv data/mut_data/GFP-train.csv \
  --guidance esm2 \
  --esm2-model facebook/esm2_t6_8M_UR50D \
  --num-samples 1000 \
  --num-chains 8 \
  --use-tess true \
  --delta 12.0 \
  --output-csv generated_seq/tfg_generation_1000.csv \
  --output-json generated_seq/tfg_generation_1000_meta.json
```

## 8 parallel chains (separate processes)

```bash
cd ML4NS-project/PLDMNEW
chmod +x run_tfg_8chains_uv.sh
bash run_tfg_8chains_uv.sh
```

This launches 8 independent ESS/TESS chains (125 samples each), then merges them into:

- `generated_seq/tfg_generation_1000.csv`

## One-command full results pipeline

```bash
cd ML4NS-project/PLDMNEW
chmod +x run_full_tfg_results_uv.sh
bash run_full_tfg_results_uv.sh
```

Environment overrides:

- `WITH_STRUCTURE=false` to skip ESMFold/TM-score stage.
- `NUM_SAMPLES=1000` and `NUM_CHAINS=8` to control generation size.

Example:

```bash
WITH_STRUCTURE=false NUM_SAMPLES=1000 NUM_CHAINS=8 bash run_full_tfg_results_uv.sh
```

Outputs are written under `results/full_analysis/`:

- `plot_umap_latent_map.png`
- `plot_identity_histogram.png`
- `plot_hamming_histogram.png`
- `plot_cross_model_correlation.png`
- `generated_with_scores.csv`
- `analysis_summary.json`

If `WITH_STRUCTURE=true`:

- `structures/structural_metrics_top100.csv`
- `structures/plot_structural_plddt_vs_identity.png`
- `structures/design_*.pdb`

## Notes

- `tfg_generation.py` handles checkpoint key formats with and without `jtae.` prefix.
- If you have a trained ESM2 fitness head, pass it via `--esm2-head-path <path>`.
- For Ada-style parallel GPU usage, launch multiple processes manually with different `CUDA_VISIBLE_DEVICES` values.