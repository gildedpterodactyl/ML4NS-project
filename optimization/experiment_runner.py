#!/usr/bin/env python3
"""
Experiment Runner: Gradient Ascent vs ESS vs TESS

Orchestrates the full optimization pipeline for all three methods:
1. Gradient Ascent  — classical gradient-based optimization
2. ESS              — Elliptical Slice Sampling (Murray et al. 2010)
3. TESS             — Transport ESS (Cabezas & Nemeth, AISTATS 2023)

Outputs feed into the protein-verification pipeline:
  - Decode optimized z → PDB
  - Score with ProteinMPNN
  - Evaluate thermal stability with ESM2StabP and MDTraj
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
        description="Run gradient ascent, ESS, and TESS optimization on random latent vectors"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to ridge_latent_tm_model.npz (trained TM oracle)",
    )
    parser.add_argument(
        "--target-tm",
        type=float,
        default=65.0,
        help="Target scalar value to optimize towards (default 65.0)",
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=100,
        help="Number of random starting vectors (default 100; up to 1000)",
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
    # TESS-specific arguments
    parser.add_argument(
        "--tess-n-transforms",
        type=int,
        default=2,
        help="Number of G-D coupling layer pairs in the TESS normalizing flow (default 2)",
    )
    parser.add_argument(
        "--tess-d-hidden",
        type=int,
        default=32,
        help="Hidden layer width for TESS coupling networks (default 32)",
    )
    parser.add_argument(
        "--tess-warmup-epochs",
        type=int,
        default=3,
        help="Number of TESS warm-up epochs h (default 3)",
    )
    parser.add_argument(
        "--tess-warmup-chains",
        type=int,
        default=10,
        help="TESS warm-up chains k per epoch (default 10)",
    )
    parser.add_argument(
        "--tess-m-grad-steps",
        type=int,
        default=10,
        help="Adam steps m per TESS NF update (default 10)",
    )
    parser.add_argument(
        "--tess-flow-lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for TESS NF training (default 1e-3)",
    )
    parser.add_argument(
        "--skip-tess",
        action="store_true",
        help="Skip TESS (run only gradient ascent and ESS, as in the original pipeline)",
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
    return parser


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
    )
    latent_dim = target_fn.get_latent_dim()
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Target value:     {args.target_tm}")
    if not args.skip_tess:
        print(f"  TESS n_transforms={args.tess_n_transforms}, "
              f"d_hidden={args.tess_d_hidden}, "
              f"warmup_epochs={args.tess_warmup_epochs}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {args.n_vectors} random starting vectors...")
    start_vectors = torch.randn(args.n_vectors, latent_dim)

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
    print(f"\nRunning {', '.join(methods)} on {args.n_vectors} vectors ({args.n_steps} steps each)...\n")

    for i in range(args.n_vectors):
        z_start = start_vectors[i]

        with torch.no_grad():
            output = target_fn(z_start)
            initial_log_score = output.log_likelihood.item()
            initial_scores.append(initial_log_score)

        print(f"Vector {i + 1}/{args.n_vectors}")
        print(f"  Initial log_score: {initial_log_score:.4f}")

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
            z_start=z_start,
            target_fn=target_fn,
            steps=args.n_steps,
        )
        z_final_ess[i] = z_final_ess_iter
        final_scores_ess.append(traj_ess[-1])

        # ── TESS ───────────────────────────────────────────────────────
        if not args.skip_tess:
            print(f"  → Running TESS...")
            z_final_tess_iter, traj_tess = run_tess(
                z_start=z_start,
                target_fn=target_fn,
                steps=args.n_steps,
                n_transforms=args.tess_n_transforms,
                d_hidden=args.tess_d_hidden,
                warmup_epochs=args.tess_warmup_epochs,
                warmup_chains=args.tess_warmup_chains,
                m_grad_steps=args.tess_m_grad_steps,
                flow_lr=args.tess_flow_lr,
            )
            z_final_tess[i] = z_final_tess_iter
            final_scores_tess.append(traj_tess[-1])

        print(f"  Results:")
        print(f"    Gradient Ascent: {traj_ga[-1]:.4f} (Δ={traj_ga[-1] - initial_log_score:+.4f})")
        print(f"    ESS:             {traj_ess[-1]:.4f} (Δ={traj_ess[-1] - initial_log_score:+.4f})")
        if not args.skip_tess:
            print(f"    TESS:            {traj_tess[-1]:.4f} (Δ={traj_tess[-1] - initial_log_score:+.4f})")
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
    def _stats(scores_final, scores_initial):
        arr_f = np.array(scores_final)
        arr_i = np.array(scores_initial)
        return {
            "initial_mean_log_score": float(np.mean(arr_i)),
            "final_mean_log_score":   float(np.mean(arr_f)),
            "final_max_log_score":    float(np.max(arr_f)),
            "final_min_log_score":    float(np.min(arr_f)),
            "final_std_log_score":    float(np.std(arr_f)),
            "mean_improvement":       float(np.mean(arr_f - arr_i)),
        }

    stats = {
        "n_vectors":  args.n_vectors,
        "n_steps":    args.n_steps,
        "latent_dim": latent_dim,
        "target_tm":  args.target_tm,
        "seed":       args.seed,
        "gradient_ascent": _stats(final_scores_gradient, initial_scores),
        "ess":             _stats(final_scores_ess,      initial_scores),
    }
    if not args.skip_tess:
        stats["tess"] = _stats(final_scores_tess, initial_scores)

    stats_json = args.output_dir / "optimization_stats.json"
    stats_json.write_text(json.dumps(stats, indent=2))
    print(f"✓ Saved statistics: {stats_json}")

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)

    def _print_method(name, s):
        print(f"\n{name}:")
        print(f"  Mean initial score:  {s['initial_mean_log_score']:8.4f}")
        print(f"  Mean final score:    {s['final_mean_log_score']:8.4f}")
        print(f"  Mean improvement:    {s['mean_improvement']:+8.4f}")
        print(f"  Best score:          {s['final_max_log_score']:8.4f}")
        print(f"  Score std:           {s['final_std_log_score']:8.4f}")

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
    print(f"\n🏆 Winner (by mean improvement): {winner}")

    print("\n" + "=" * 70)
    print("NEXT STEPS: Validation Pipeline")
    print("=" * 70)
    print("""
Run the Rg pipeline for all three methods:
  $ bash runner/run_rg_pipeline.sh

Or submit to SLURM:
  $ sbatch runner/submit_rg_pipeline.slurm

The pipeline will decode z_final_tess.pt → PDB → Rg and compare
radius of gyration distributions across all three methods.
""")


if __name__ == "__main__":
    main()
