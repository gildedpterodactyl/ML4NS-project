# Available Datasets

All datasets live under `PLDM_COMPARISON_SUITE/data/mut_data/` and mirrored
in `../PROLDM_OUTLIER/data/mut_data/`.

## Currently available

| Dataset | Task | Fitness column | WT PDB | Notes |
|---------|------|---------------|--------|-------|
| **GFP** | avGFP fluorescence brightness | `log_fluorescence` | `1GFL` | Sarkisyan et al. 2016; ~54k single+multi mutants |
| **TAPE** | Stability (Rocklin et al.) | `stability_score` / `tm` | none | No WT crystal structure — structure metrics skipped |

## Mode taxonomy

The `--mode` flag in `generate_sequences.py` selects one of **four** distinct methods:

| Mode | Algorithm | Init | `--use-transport` | What it is |
|------|-----------|------|-------------------|------------|
| `baseline` | PRO-LDM diffusion sampler | label-conditioned Gaussian noise | N/A | **Paper's own method** — no MCMC, just diffusion |
| `ess` | Elliptical Slice Sampling | Top-fitness training seq | `false` | Plain ESS from best known sequence |
| `tess` | Elliptical Slice Sampling | WT centroid | `false` | Same ESS algorithm, different starting point |
| `transport_ess` | ESS + learned RealNVP flow preconditioning | Top-fitness training seq | `true` | **True Transport ESS** — the novel contribution |

> **Important:** `--mode tess` is NOT Transport ESS. It is plain ESS
> initialised from the WT sequence rather than the highest-fitness sequence.
> Real Transport ESS requires `--mode transport_ess` (which internally sets
> `use_transport=True` and activates the RealNVP flow in `sampler_ess.py`).

## Adding a new dataset

1. Add `<NAME>-train.csv` and `<NAME>-test.csv` to `data/mut_data/`.
   Required columns:
   - Sequence: any of `sequence`, `sequence_trimmed`, `pred_seq`, `primary`
   - Fitness: any of `fitness`, `log_fluorescence`, `tm`, `stability_score`, `enrichment`

2. Train a PRO-LDM AE checkpoint: `../PROLDM_OUTLIER/train_logs/<NAME>/epoch_1000.pt`

3. Train a baseline diffusion checkpoint: `../PROLDM_OUTLIER/train_logs/<NAME>/dropout_tiny_epoch_1000.pt`

4. Add a row to `DATASET_REGISTRY` in `run_all_datasets.sh`:
   ```bash
   "MYDATA:data/mut_data/MYDATA-train.csv:train_logs/MYDATA/epoch_1000.pt:train_logs/MYDATA/dropout_tiny_epoch_1000.pt:PDBID"
   ```

5. Run: `DATASETS="MYDATA" bash run_all_datasets.sh`

## Fitness column auto-detection

`pipeline_utils.infer_fitness_col()` scans in order:
```
fitness -> log_fluorescence -> tm -> enrichment -> stability_score
```

## Interpreting the summary table

| Metric | Good direction | What it means |
|--------|---------------|---------------|
| `mean_internal_regressor_score` | ↑ higher | Generated seqs predicted more fit than training mean |
| `mean_positional_kl_divergence` | ↓ lower | Generated AA distribution stays close to training |
| `loo_spearman_rho` | ↑ closer to 1.0 | Regressor is well-calibrated on known sequences |
| `spearman_rho_vs_truth` | ↑ closer to 1.0 | Regressor ranks near-training seqs correctly vs. true DMS |
| `mean_nearest_train_identity` | neither extreme | ≥0.99 = memorised; ≤0.5 = implausible |
| `mean_plddt_score` | ↑ above 70 | ESMFold confident in predicted structure |
| `ess_per_sec` | ↑ higher | Chain mixes faster (transport_ess should beat ess/tess) |
