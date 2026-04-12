#!/usr/bin/env python3
"""
Experiment Runner: Gradient Ascent vs ESS vs TESS

Orchestrates the full optimization pipeline for all three methods:
1. Gradient Ascent  — classical gradient-based optimization
2. ESS              — Elliptical Slice Sampling (Murray et al. 2010)
3. TESS             — Transport ESS (Cabezas & Nemeth, AISTATS 2023)

Critical fix: training-prior sampling
-------------------------------------
The oracle was trained on a collapsed posterior: all training latents
lie in a tight cluster at x_mean ≈ [-0.83, 0.36, 0.59, 0.84, ...] with
x_std ≈ [0.08-0.29]. The model prior N(0,I) is 3-6σ away in every dim.

Using z ~ N(0,I) as start vectors causes:
  - Decoder gets OOD inputs  → degenerate PDB → Rg ≈ 3-5 Å always
  - Oracle predicts 50.6 Å at z=0 (should be ~26.8 Å = intercept)
  - Oracle range for N(0,I): 48 ± 74 Å → physically impossible values

Fix: always sample from N(x_mean, diag(x_std²)) — the training distribution.

Temperature guidance
--------------------
With start vectors now in the correct region, score at z_start ≈ intercept.
Rule of thumb: T ≈ (intercept - target)^2 / 4
  target=5  Å → T ≈ (26.8-5)^2/4  ≈ 119  → use --temperature 120
  target=20 Å → T ≈ (26.8-20)^2/4 ≈  12  → use --temperature 12
  target=30 Å → T ≈ (30-26.8)^2/4 ≈   3  → use --temperature 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import random


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run gradient ascent, ESS, and TESS optimization on random latent vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to ridge_latent_*_model.npz (trained oracle)",
    )
    parser.add_argument(
        "--target-tm",
        type=float,
        default=65.0,
        help="Target scalar value to optimize towards",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=100.0,
        help=(
            "Likelihood temperature T: score = -(pred-target)^2 / T.  "
            "Rule of thumb: T ≈ (oracle_intercept - target)^2 / 4.  "
            "With intercept≈26.8 Å: target=5→T≈120, target=20→T≈12, "
            "target=30→T≈3.  Default 100.0 is safe for most Rg targets."
        ),
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=100,
        help="Number of random starting vectors",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of optimization steps per vector",
    )
    parser.add_argument(
        "--gradient-lr",
        type=float,
        default=0.1,
        help="Learning rate for gradient ascent",
    )
    parser.add_argument(
        "--gradient-penalty",
        type=float,
        default=0.05,
        help="L2 penalty weight for gradient ascent",
    )
    # ESS options
    parser.add_argument(
        "--ess-warm-start",
        action="store_true",
        default=True,
        help=(
            "Warm-start ESS/TESS: run 20 gradient-ascent steps before the MCMC "
            "chain so the starting point is already in a moderate-score region."
        ),
    )
    parser.add_argument(
        "--no-ess-warm-start",
        dest="ess_warm_start",
        action="store_false",
        help="Disable ESS/TESS warm start (start MCMC from raw z_start)",
    )
    # TESS-specific arguments
    parser.add_argument(
        "--tess-n-transforms",
        type=int,
        default=2,
        help="Number of G-D coupling layer pairs in the TESS normalizing flow",
    )
    parser.add_argument(
        "--tess-d-hidden",
        type=int,
        default=32,
        help="Hidden layer width for TESS coupling networks",
    )
    parser.add_argument(
        "--tess-warmup-epochs",
        type=int,
        default=3,
        help="Number of TESS warm-up epochs h",
    )
    parser.add_argument(
        "--tess-warmup-chains",
        type=int,
        default=10,
        help="TESS warm-up chains k per epoch",
    )
    parser.add_argument(
        "--tess-m-grad-steps",
        type=int,
        default=10,
        help="Adam steps m per TESS NF update",
    )
    parser.add_argument(
        "--tess-flow-lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for TESS NF training",
    )
    parser.add_argument(
        "--skip-tess",
        action="store_true",
        help="Skip TESS (run only gradient ascent and ESS)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optimization/outputs"),
        help="Directory to save trajectories and optimized vectors",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--retrain-oracle",
        action="store_true",
        help=(
            "Generate oracle retraining data: sample 500 z vectors from the "
            "training prior N(x_mean, x_std^2), evaluate oracle-predicted Rg "
            "for each, and save retrain_oracle_data.npz to --output-dir. "
            "Use this data to retrain the ridge oracle on the actual decoder domain."
        ),
    )
    return parser


def _warm_start_z(
    z_start: torch.Tensor,
    target_fn,
    warm_steps: int = 20,
    lr: float = 0.1,
) -> torch.Tensor:
    """
    Run a short gradient ascent (no penalty) to move z_start into a
    region of non-trivial score before starting the MCMC chain.
    """
    z = z_start.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(warm_steps):
        opt.zero_grad()
        loss = -target_fn(z).log_likelihood
        loss.backward()
        opt.step()
    return z.detach()


def _load_prior_params(checkpoint_path: Path):
    """Load x_mean and x_std from oracle checkpoint for training-prior sampling."""
    with np.load(checkpoint_path) as d:
        x_mean = torch.tensor(d["x_mean"], dtype=torch.float32)
        x_std  = torch.tensor(d["x_std"],  dtype=torch.float32)
        intercept = float(d["intercept"].flat[0])
        coef = d["coef"]
    return x_mean, x_std, intercept, coef


def _sample_from_training_prior(
    n: int,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample n vectors from the training distribution N(x_mean, diag(x_std^2)).

    CRITICAL: Do NOT use torch.randn(n, d) — that samples from N(0,I) which is
    3-6σ away from the oracle's training domain in every dimension.
    The decoder maps N(0,I) inputs to degenerate proteins with Rg ≈ 3-5 Å.
    """
    if seed is not None:
        torch.manual_seed(seed)
    eps = torch.randn(n, len(x_mean))
    return x_mean.unsqueeze(0) + eps * x_std.unsqueeze(0)


def main() -> None:
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from optimization.target_function import TargetFunction
    from optimization.gradient_ascent import run_gradient_ascent
    from optimization.elliptical_slice_sampling import run_ess
    from optimization.transport_ess import run_tess

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading Oracle from {args.checkpoint}...")
    target_fn = TargetFunction.from_checkpoint(
        model_path=args.checkpoint,
        target_value=args.target_tm,
        temperature=args.temperature,
    )
    latent_dim = target_fn.get_latent_dim()

    # Load prior params from oracle checkpoint
    x_mean, x_std, _intercept, _coef = _load_prior_params(args.checkpoint)
    suggested_T = (_intercept - args.target_tm) ** 2 / 4

    print(f"  Latent dimension:      {latent_dim}")
    print(f"  Oracle intercept:      {_intercept:.2f} Å")
    print(f"  Training prior mean:   {x_mean.numpy().round(4)}")
    print(f"  Training prior std:    {x_std.numpy().round(4)}")
    print(f"  Prior range (±3σ):     [{(x_mean - 3*x_std).numpy().round(2)}")
    print(f"                          {(x_mean + 3*x_std).numpy().round(2)}]")
    print(f"  z=0 distance from prior: {((torch.zeros(latent_dim) - x_mean) / x_std).abs().numpy().round(1)} σ")
    print(f"  Target value:          {args.target_tm}")
    print(f"  Temperature used:      {args.temperature}")
    print(f"  Suggested T:           {suggested_T:.1f}")
    if abs(args.temperature - suggested_T) / max(suggested_T, 1) > 2:
        print(f"  [WARN] Temperature {args.temperature} differs significantly from "
              f"suggested {suggested_T:.1f}. ESS/TESS mixing may be poor.")
    if not args.skip_tess:
        print(f"  TESS n_transforms={args.tess_n_transforms}, "
              f"d_hidden={args.tess_d_hidden}, "
              f"warmup_epochs={args.tess_warmup_epochs}")
    print(f"  ESS/TESS warm_start:   {args.ess_warm_start}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Optional: generate oracle retraining data
    # ------------------------------------------------------------------
    if args.retrain_oracle:
        print("\n" + "=" * 60)
        print("GENERATING ORACLE RETRAINING DATA")
        print("=" * 60)
        n_retrain = 500
        print(f"  Sampling {n_retrain} z vectors from training prior N(x_mean, x_std²)...")
        z_retrain = _sample_from_training_prior(n_retrain, x_mean, x_std, seed=args.seed)
        pred_rgs = []
        with torch.no_grad():
            for j in range(n_retrain):
                out = target_fn(z_retrain[j])
                pred_rgs.append(out.pred_value.item())
        pred_rgs = np.array(pred_rgs)
        out_npz = args.output_dir / "retrain_oracle_data.npz"
        np.savez(
            out_npz,
            z=z_retrain.numpy(),
            pred_rg=pred_rgs,
            x_mean=x_mean.numpy(),
            x_std=x_std.numpy(),
        )
        print(f"  Oracle-predicted Rg: {pred_rgs.mean():.2f} ± {pred_rgs.std():.2f} Å  "
              f"[{pred_rgs.min():.2f}, {pred_rgs.max():.2f}]")
        print(f"  Saved: {out_npz}")
        print("")
        print("  NEXT STEP: decode these z vectors to PDB, measure actual Rg,")
        print("  then retrain ridge on (z → actual_Rg) using regression/train_oracle.py")
        print("="*60 + "\n")

    # ------------------------------------------------------------------
    # Generate start vectors from TRAINING PRIOR, not N(0,I)
    # ------------------------------------------------------------------
    print(f"\nGenerating {args.n_vectors} start vectors from training prior "
          f"N(x_mean, x_std²)...")
    start_vectors = _sample_from_training_prior(args.n_vectors, x_mean, x_std, seed=args.seed)

    # Sanity check: oracle should predict ~intercept ± a few Å
    with torch.no_grad():
        sample_preds = [target_fn(start_vectors[j]).pred_value.item()
                        for j in range(min(20, args.n_vectors))]
    sample_preds = np.array(sample_preds)
    print(f"  Oracle-predicted Rg at start vectors: "
          f"{sample_preds.mean():.2f} ± {sample_preds.std():.2f} Å  "
          f"[{sample_preds.min():.2f}, {sample_preds.max():.2f}]")
    print(f"  (Should be ≈ {_intercept:.1f} ± a few Å — if wildly off, retrain oracle)")

    all_trajectories = {
        "vector_id": [],
        "step": [],
        "method": [],
        "log_score": [],
    }

    z_final_gradient = torch.zeros(args.n_vectors, latent_dim)
    z_final_ess      = torch.zeros(args.n_vectors, latent_dim)
    z_final_tess     = torch.zeros(args.n_vectors, latent_dim)

    initial_scores        = []
    final_scores_gradient = []
    final_scores_ess      = []
    final_scores_tess     = []

    methods = ["gradient_ascent", "ess"] + ([] if args.skip_tess else ["tess"])
    print(f"\nRunning {', '.join(methods)} on {args.n_vectors} vectors "
          f"({args.n_steps} steps each)...\n")

    for i in range(args.n_vectors):
        z_start = start_vectors[i]

        with torch.no_grad():
            output = target_fn(z_start)
            initial_log_score = output.log_likelihood.item()
            initial_scores.append(initial_log_score)

        print(f"Vector {i + 1}/{args.n_vectors}")
        print(f"  Initial log_score: {initial_log_score:.4f}  "
              f"(pred_rg={output.pred_value.item():.2f} Å)")

        # Warm-start for MCMC methods
        if args.ess_warm_start:
            z_mcmc_start = _warm_start_z(z_start, target_fn)
            with torch.no_grad():
                ws_out = target_fn(z_mcmc_start)
            print(f"  Warm-start log_score: {ws_out.log_likelihood.item():.4f}  "
                  f"(pred_rg={ws_out.pred_value.item():.2f} Å)")
        else:
            z_mcmc_start = z_start

        # ── Gradient Ascent ────────────────────────────────────────────
        print(f"  → Running Gradient Ascent...")
        z_final_ga, traj_ga = run_gradient_ascent(
            z_start=z_start,
            target_fn=target_fn,
            steps=args.n_steps,
            lr=args.gradient_lr,
            penalty_weight=args.gradient_penalty,
        )
        z_final_gradient[i] = z_final_ga
        final_scores_gradient.append(traj_ga[-1])

        # ── ESS ────────────────────────────────────────────────────────
        print(f"  → Running ESS...")
        z_final_ess_iter, traj_ess = run_ess(
            z_start=z_mcmc_start,
            target_fn=target_fn,
            steps=args.n_steps,
            x_mean=x_mean,
            x_std=x_std,
        )
        z_final_ess[i] = z_final_ess_iter
        final_scores_ess.append(traj_ess[-1])

        # ── TESS ───────────────────────────────────────────────────────
        if not args.skip_tess:
            print(f"  → Running TESS...")
            z_final_tess_iter, traj_tess = run_tess(
                z_start=z_mcmc_start,
                target_fn=target_fn,
                steps=args.n_steps,
                n_transforms=args.tess_n_transforms,
                d_hidden=args.tess_d_hidden,
                warmup_epochs=args.tess_warmup_epochs,
                warmup_chains=args.tess_warmup_chains,
                m_grad_steps=args.tess_m_grad_steps,
                flow_lr=args.tess_flow_lr,
                x_mean=x_mean,
                x_std=x_std,
            )
            z_final_tess[i] = z_final_tess_iter
            final_scores_tess.append(traj_tess[-1])

        print(f"  Results:")
        with torch.no_grad():
            ga_pred  = target_fn(z_final_ga).pred_value.item()
            ess_pred = target_fn(z_final_ess_iter).pred_value.item()
        print(f"    Gradient Ascent: score={traj_ga[-1]:.4f}  pred_rg={ga_pred:.2f} Å  "
              f"(Δ={traj_ga[-1] - initial_log_score:+.4f})")
        print(f"    ESS:             score={traj_ess[-1]:.4f}  pred_rg={ess_pred:.2f} Å  "
              f"(Δ={traj_ess[-1] - initial_log_score:+.4f})")
        if not args.skip_tess:
            with torch.no_grad():
                tess_pred = target_fn(z_final_tess_iter).pred_value.item()
            print(f"    TESS:            score={traj_tess[-1]:.4f}  pred_rg={tess_pred:.2f} Å  "
                  f"(Δ={traj_tess[-1] - initial_log_score:+.4f})")
        print()

        for step, score_ga in enumerate(traj_ga):
            all_trajectories["vector_id"].append(i)
            all_trajectories["step"].append(step)
            all_trajectories["method"].append("gradient_ascent")
            all_trajectories["log_score"].append(score_ga)

        for step, score_ess in enumerate(traj_ess):
            all_trajectories["vector_id"].append(i)
            all_trajectories["step"].append(step)
            all_trajectories["method"].append("ess")
            all_trajectories["log_score"].append(score_ess)

        if not args.skip_tess:
            for step, score_tess in enumerate(traj_tess):
                all_trajectories["vector_id"].append(i)
                all_trajectories["step"].append(step)
                all_trajectories["method"].append("tess")
                all_trajectories["log_score"].append(score_tess)

    # Save trajectories
    traj_df = pd.DataFrame(all_trajectories)
    traj_csv = args.output_dir / "trajectories.csv"
    traj_df.to_csv(traj_csv, index=False)
    print(f"\n✓ Saved trajectories: {traj_csv}")

    # Save optimized vectors
    torch.save(z_final_gradient, args.output_dir / "z_final_gradient_ascent.pt")
    torch.save(z_final_ess,      args.output_dir / "z_final_ess.pt")
    if not args.skip_tess:
        torch.save(z_final_tess, args.output_dir / "z_final_tess.pt")
    print(f"✓ Saved optimized vectors in {args.output_dir}")

    # Compute statistics
    def _stats(scores_final, scores_initial, z_finals):
        arr_f = np.array(scores_final)
        arr_i = np.array(scores_initial)
        with torch.no_grad():
            preds = np.array([target_fn(z_finals[j]).pred_value.item()
                              for j in range(len(z_finals))])
        return {
            "initial_mean_log_score":  float(np.mean(arr_i)),
            "final_mean_log_score":    float(np.mean(arr_f)),
            "final_max_log_score":     float(np.max(arr_f)),
            "final_min_log_score":     float(np.min(arr_f)),
            "final_std_log_score":     float(np.std(arr_f)),
            "mean_improvement":        float(np.mean(arr_f - arr_i)),
            "pred_rg_mean":            float(np.mean(preds)),
            "pred_rg_std":             float(np.std(preds)),
            "pred_rg_min":             float(np.min(preds)),
            "pred_rg_max":             float(np.max(preds)),
        }

    stats = {
        "n_vectors":   args.n_vectors,
        "n_steps":     args.n_steps,
        "latent_dim":  latent_dim,
        "target_tm":   args.target_tm,
        "temperature": args.temperature,
        "seed":        args.seed,
        "oracle_intercept": _intercept,
        "gradient_ascent": _stats(final_scores_gradient, initial_scores, z_final_gradient),
        "ess":             _stats(final_scores_ess,      initial_scores, z_final_ess),
    }
    if not args.skip_tess:
        stats["tess"] = _stats(final_scores_tess, initial_scores, z_final_tess)

    stats_json = args.output_dir / "optimization_stats.json"
    stats_json.write_text(json.dumps(stats, indent=2))
    print(f"✓ Saved statistics: {stats_json}")

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)

    def _print_method(name, s):
        print(f"\n{name}:")
        print(f"  Mean initial score:   {s['initial_mean_log_score']:8.4f}")
        print(f"  Mean final score:     {s['final_mean_log_score']:8.4f}")
        print(f"  Mean improvement:     {s['mean_improvement']:+8.4f}")
        print(f"  Best score:           {s['final_max_log_score']:8.4f}")
        print(f"  Score std:            {s['final_std_log_score']:8.4f}")
        print(f"  Pred Rg mean ± std:   {s['pred_rg_mean']:.2f} ± {s['pred_rg_std']:.2f} Å  "
              f"[{s['pred_rg_min']:.2f}, {s['pred_rg_max']:.2f}]")

    _print_method("Gradient Ascent", stats["gradient_ascent"])
    _print_method("Elliptical Slice Sampling (ESS)", stats["ess"])
    if not args.skip_tess:
        _print_method("Transport ESS (TESS)", stats["tess"])

    # Determine winner
    candidates = {
        "Gradient Ascent": stats["gradient_ascent"]["mean_improvement"],
        "ESS":             stats["ess"]["mean_improvement"],
    }
    if not args.skip_tess:
        candidates["TESS"] = stats["tess"]["mean_improvement"]
    winner = max(candidates, key=candidates.get)
    print(f"\n\U0001f3c6 Winner (by mean improvement): {winner}")

    print("\n" + "=" * 70)
    print("NEXT STEPS: Validation Pipeline")
    print("=" * 70)
    print(f"""
Run the Rg pipeline with the correct temperature:
  $ bash runner/run_rg_pipeline.sh {args.target_tm} {args.n_vectors} \\
        {args.n_steps} {args.temperature}

To generate oracle retraining data (recommended):
  $ python -m optimization.experiment_runner \\
      --checkpoint {args.checkpoint} \\
      --target-tm {args.target_tm} \\
      --n-vectors 1 --n-steps 1 \\
      --retrain-oracle \\
      --output-dir {args.output_dir}
  Then decode retrain_oracle_data.npz z-vectors to PDB,
  measure actual Rg, and retrain with regression/train_oracle.py
""")


if __name__ == "__main__":
    main()
