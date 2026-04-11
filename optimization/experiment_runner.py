#!/usr/bin/env python3
"""
Experiment Runner: The Oracle Hack Challenge

This is the master script that orchestrates the entire optimization pipeline:
1. Generates starting populations
2. Runs both algorithms on each member
3. Logs trajectories for analysis
4. Saves optimized latent vectors for downstream validation

The outputs feed directly into the protein-verification pipeline:
  - Decode optimized z → PDB
  - Score with ProteinMPNN
  - Evaluate with ESM2StabP and MDTraj
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
        description="Run gradient ascent and ESS optimization on random latent vectors"
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

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load target function
    from optimization.target_function import TargetFunction
    from optimization.gradient_ascent import run_gradient_ascent
    from optimization.elliptical_slice_sampling import run_ess

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading Oracle from {args.checkpoint}...")
    target_fn = TargetFunction.from_checkpoint(
        model_path=args.checkpoint,
        target_value=args.target_tm,
    )
    latent_dim = target_fn.get_latent_dim()
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Target value: {args.target_tm}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate random starting vectors
    print(f"\nGenerating {args.n_vectors} random starting vectors...")
    start_vectors = torch.randn(args.n_vectors, latent_dim)

    # Initialize result containers
    all_trajectories = {
        "vector_id": [],
        "step": [],
        "method": [],
        "log_score": [],
    }

    z_final_gradient = torch.zeros(args.n_vectors, latent_dim)
    z_final_ess = torch.zeros(args.n_vectors, latent_dim)

    initial_scores = []
    final_scores_gradient = []
    final_scores_ess = []

    # Run optimization for each starting vector
    print(f"\nRunning optimization on {args.n_vectors} vectors ({args.n_steps} steps each)...\n")

    for i in range(args.n_vectors):
        z_start = start_vectors[i]

        # Initial score
        with torch.no_grad():
            output = target_fn(z_start)
            initial_log_score = output.log_likelihood.item()
            initial_scores.append(initial_log_score)

        print(f"Vector {i + 1}/{args.n_vectors}")
        print(f"  Initial log_score: {initial_log_score:.4f}")

        # Run Gradient Ascent
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

        # Run ESS
        print(f"  → Running ESS...")
        z_final_ess_iter, traj_ess = run_ess(
            z_start=z_start,
            target_fn=target_fn,
            steps=args.n_steps,
        )
        z_final_ess[i] = z_final_ess_iter
        final_scores_ess.append(traj_ess[-1])

        print(f"  Results:")
        print(f"    Gradient Ascent: {traj_ga[-1]:.4f} (Δ={traj_ga[-1] - initial_log_score:+.4f})")
        print(f"    ESS:             {traj_ess[-1]:.4f} (Δ={traj_ess[-1] - initial_log_score:+.4f})")
        print()

        # Collect trajectories
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

    # Save trajectory data
    traj_df = pd.DataFrame(all_trajectories)
    traj_csv = args.output_dir / "trajectories.csv"
    traj_df.to_csv(traj_csv, index=False)
    print(f"\n✓ Saved trajectories: {traj_csv}")

    # Save optimized vectors
    torch.save(
        z_final_gradient,
        args.output_dir / "z_final_gradient_ascent.pt",
    )
    torch.save(
        z_final_ess,
        args.output_dir / "z_final_ess.pt",
    )
    print(f"✓ Saved optimized vectors:")
    print(f"    Gradient Ascent: {args.output_dir / 'z_final_gradient_ascent.pt'}")
    print(f"    ESS:             {args.output_dir / 'z_final_ess.pt'}")

    # Compute and save statistics
    stats = {
        "n_vectors": args.n_vectors,
        "n_steps": args.n_steps,
        "latent_dim": latent_dim,
        "target_tm": args.target_tm,
        "seed": args.seed,
        "gradient_ascent": {
            "initial_mean_log_score": float(np.mean(initial_scores)),
            "final_mean_log_score": float(np.mean(final_scores_gradient)),
            "final_max_log_score": float(np.max(final_scores_gradient)),
            "final_min_log_score": float(np.min(final_scores_gradient)),
            "final_std_log_score": float(np.std(final_scores_gradient)),
            "mean_improvement": float(
                np.mean(np.array(final_scores_gradient) - np.array(initial_scores))
            ),
        },
        "ess": {
            "initial_mean_log_score": float(np.mean(initial_scores)),
            "final_mean_log_score": float(np.mean(final_scores_ess)),
            "final_max_log_score": float(np.max(final_scores_ess)),
            "final_min_log_score": float(np.min(final_scores_ess)),
            "final_std_log_score": float(np.std(final_scores_ess)),
            "mean_improvement": float(
                np.mean(np.array(final_scores_ess) - np.array(initial_scores))
            ),
        },
    }

    stats_json = args.output_dir / "optimization_stats.json"
    stats_json.write_text(json.dumps(stats, indent=2))
    print(f"✓ Saved statistics: {stats_json}")

    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"\nGradient Ascent:")
    print(
        f"  Mean initial score:    {stats['gradient_ascent']['initial_mean_log_score']:8.4f}"
    )
    print(
        f"  Mean final score:      {stats['gradient_ascent']['final_mean_log_score']:8.4f}"
    )
    print(
        f"  Mean improvement:      {stats['gradient_ascent']['mean_improvement']:+8.4f}"
    )
    print(f"  Best score:            {stats['gradient_ascent']['final_max_log_score']:8.4f}")

    print(f"\nElliptical Slice Sampling:")
    print(
        f"  Mean initial score:    {stats['ess']['initial_mean_log_score']:8.4f}"
    )
    print(f"  Mean final score:      {stats['ess']['final_mean_log_score']:8.4f}")
    print(f"  Mean improvement:      {stats['ess']['mean_improvement']:+8.4f}")
    print(f"  Best score:            {stats['ess']['final_max_log_score']:8.4f}")

    winner = (
        "ESS"
        if stats["ess"]["mean_improvement"] > stats["gradient_ascent"]["mean_improvement"]
        else "Gradient Ascent"
    )
    print(f"\n🏆 Winner (by mean improvement): {winner}")

    print("\n" + "=" * 70)
    print("NEXT STEPS: Validation Pipeline")
    print("=" * 70)
    print(f"""
The optimized latent vectors are ready for downstream validation:

1. Decode optimized z vectors back to protein structures:
   $ python validation/decode_and_validate.py \\
       --z-vectors optimization/outputs/z_final_ess.pt \\
       --decoder-checkpoint <path/to/ae.ckpt> \\
       --output-dir validation/outputs

2. Score with ProteinMPNN (sequence design):
   $ python validation/score_with_proteinmpnn.py \\
       --pdbs validation/outputs/pdbs \\
       --output-dir validation/outputs

3. Evaluate thermal stability with ESM2StabP:
   $ python validation/predict_thermal_stability.py \\
       --sequences validation/outputs/sequences \\
       --output-dir validation/outputs

4. Compare metrics: original vs optimized proteins
   $ python validation/compare_metrics.py \\
       --original-metrics regression/outputs/test_predictions.csv \\
       --optimized-metrics validation/outputs/metrics.csv

See OPTIMIZATION_INSTRUCTIONS.md for detailed usage examples.
""")


if __name__ == "__main__":
    main()
