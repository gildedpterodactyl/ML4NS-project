# Available Datasets

All datasets live under `PLDM_COMPARISON_SUITE/data/mut_data/` and mirrored
in `../PROLDM_OUTLIER/data/mut_data/`. The train split is used for novelty
metrics and LOO regressor calibration; the test split is used for the
Spearman-ρ vs truth check.

## Currently available

| Dataset | Task | Fitness column | WT PDB | Notes |
|---------|------|---------------|--------|-------|
| **GFP** | avGFP fluorescence brightness | `log_fluorescence` | `1GFL` | Primary benchmark from Sarkisyan et al. 2016; ~54k single+multi mutants |
| **TAPE** | Stability (Rocklin et al.) | `stability_score` / `tm` | none | Thermostability; no WT crystal structure — structure metrics skipped |

## Adding a new dataset

1. Add `<NAME>-train.csv` and `<NAME>-test.csv` to `data/mut_data/`.
   Required columns:
   - A sequence column: any of `sequence`, `sequence_trimmed`, `pred_seq`, `primary`
   - A fitness column: any of `fitness`, `log_fluorescence`, `tm`, `stability_score`, `enrichment`

2. Train a PRO-LDM checkpoint on your dataset and place it at
   `../PROLDM_OUTLIER/train_logs/<NAME>/epoch_1000.pt`.

3. Add one row to `DATASET_REGISTRY` in `run_all_datasets.sh`:
   ```bash
   "MYDATA:data/mut_data/MYDATA-train.csv:train_logs/MYDATA/epoch_1000.pt:PDBID"
   ```
   Use `none` as the PDB ID if no WT structure is available.

4. Run:
   ```bash
   DATASETS="MYDATA" bash run_all_datasets.sh
   ```

## Fitness column auto-detection

`pipeline_utils.infer_fitness_col()` scans for these column names in order:
```
fitness  ->  log_fluorescence  ->  tm  ->  enrichment  ->  stability_score
```
If your column has a different name, add it to that function.

## Interpreting the summary table

| Metric | Good direction | What it means |
|--------|---------------|---------------|
| `mean_internal_regressor_score` | ↑ higher | Generated seqs predicted more fit than training mean |
| `mean_positional_kl_divergence` | ↓ lower | Generated AA distribution stays close to training |
| `loo_spearman_rho` | ↑ closer to 1.0 | Regressor is well-calibrated on known sequences |
| `spearman_rho_vs_truth` | ↑ closer to 1.0 | Regressor correctly ranks near-training seqs vs. true DMS |
| `mean_nearest_train_identity` | neither extreme | Too high (≥0.99) = memorised; too low (≤0.5) = implausible |
| `mean_plddt_score` | ↑ above 70 | ESMFold is confident in predicted structure |
| `ess_per_sec` | ↑ higher | MCMC chain mixes faster (TESS should beat ESS here) |
