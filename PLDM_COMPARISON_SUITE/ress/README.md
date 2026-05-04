# RESS — Restart Elliptical Slice Sampling

RESS is an extension of ESS that uses **iterative fitness-guided restarts**
to escape stuck chains and explore multiple fitness peaks in the latent space.

## Why plain ESS gets stuck

A single ESS chain initialised near the highest-fitness training sequence
quickly exhausts the unique sequences reachable from that seed under the
current bracket budget.  After ~6–20 unique acceptances the chain starts
returning the same latent point repeatedly (max-iteration bracket collapse),
filling the requested N with duplicates.

## What RESS does differently

```
Round 0
  ├─ Encode top-K diverse training seqs as initial latents
  └─ Run num_chains ESS chains for max_steps each
       → collect all unique (sequence, latent, score) triples

Round r  (r = 1 … ress_rounds)
  ├─ From all accepted latents so far, select top-K by score
  │    filtered so no two seeds are within min_hamming_diversity
  │    of each other (prevents all chains collapsing into one basin)
  ├─ Slightly widen temperature: T *= restart_temp_scale  (default 1.05)
  └─ Run num_chains chains seeded from the top-K latents
       → add newly seen sequences to the global pool

Stop when
  new_unique_this_round < patience  OR  round == ress_rounds

Output
  Top-N sequences by score, globally deduplicated across all rounds.
```

## Usage

```bash
cd PLDM_COMPARISON_SUITE

# Basic (CPU)
bash ress/run_ress.sh

# GPU, 1000 sequences
N=1000 DEVICE=cuda bash ress/run_ress.sh

# Standalone
uv run python ress/generate_ress.py \
  --proldm-root ../PROLDM_OUTLIER \
  --train-csv   data/mut_data/GFP-train.csv \
  --ae-ckpt     train_logs/GFP/epoch_1000.pt \
  --dataset     GFP \
  --n           1000 \
  --num-chains  8 \
  --ress-rounds 5 \
  --ress-top-k  8 \
  --ress-min-hamming 3 \
  --ress-patience    50 \
  --ress-temp-scale  1.05 \
  --latent-temperature 1.0 \
  --tess-delta  0.5 \
  --burnin      100 \
  --max-steps   8000 \
  --use-transport false \
  --device cuda \
  --out-dir ress/outputs/GFP
```

## Output files

| File | Contents |
|------|---------|
| `ress/outputs/<DS>/results_ress.csv` | Per-sequence: sequence, score, chain, step, round, restart_seed_seq |
| `ress/outputs/<DS>/ress_round_log.csv` | Per-round: round, n_new, n_total, best_score, temperature |

## Key parameters

| Flag | Default | Effect |
|------|---------|--------|
| `--ress-rounds` | 5 | Max number of restart rounds |
| `--ress-top-k` | 8 | Seeds selected per round |
| `--ress-min-hamming` | 3 | Min AA Hamming between seeds (diversity filter) |
| `--ress-patience` | 50 | Stop early if fewer than this many new seqs found |
| `--ress-temp-scale` | 1.05 | Temperature multiplier per round (slight widening) |
| `--latent-temperature` | 1.0 | Base ESS temperature |
| `--tess-delta` | 0.5 | Ball radius around seed (set larger than plain ESS) |
| `--use-transport` | false | If true, each chain uses TransportESSSampler (RTESS) |

## Relationship to ESS / Transport ESS

- `--use-transport false` → **RESS**: restart ESS
- `--use-transport true`  → **RTESS**: restart Transport ESS (RealNVP flow + restarts)

Both share the same restart logic; transport only changes the per-step sampler.
