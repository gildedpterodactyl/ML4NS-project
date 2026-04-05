#!/usr/bin/env python3
"""
Plot Convergence Trajectories from Optimization Experiments

Generates publication-quality plots comparing Gradient Ascent vs ESS convergence.

Usage:
    python optimization/plot_trajectories.py \
        --results-dir optimization/outputs \
        --output-dir plots

Creates:
    - convergence_summary.png (mean ± std curves)
    - convergence_all_vectors.png (all 100 trajectories)
    - algorithm_comparison.png (Algorithm 1 vs Algorithm 2 boxplot)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot optimization trajectories")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("optimization/outputs"),
        help="Directory containing trajectories.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save plots",
    )
    return parser


def main():
    args = build_parser().parse_args()

    # Load data
    traj_file = args.results_dir / "trajectories.csv"
    if not traj_file.exists():
        print(f"Error: {traj_file} not found")
        return

    df = pd.read_csv(traj_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_vectors = df["vector_id"].max() + 1
    n_steps = df["step"].max() + 1

    # Split by method
    df_ga = df[df["method"] == "gradient_ascent"]
    df_ess = df[df["method"] == "ess"]

    # ========== Plot 1: Mean ± Std Convergence ==========
    fig, ax = plt.subplots(figsize=(12, 6))

    # Gradient Ascent
    ga_scores = df_ga.groupby("step")["log_score"].agg(["mean", "std"])
    ax.plot(
        ga_scores.index,
        ga_scores["mean"],
        "o-",
        label="Gradient Ascent",
        linewidth=2,
        markersize=5,
    )
    ax.fill_between(
        ga_scores.index,
        ga_scores["mean"] - ga_scores["std"],
        ga_scores["mean"] + ga_scores["std"],
        alpha=0.3,
    )

    # ESS
    ess_scores = df_ess.groupby("step")["log_score"].agg(["mean", "std"])
    ax.plot(
        ess_scores.index,
        ess_scores["mean"],
        "s-",
        label="Elliptical Slice Sampling",
        linewidth=2,
        markersize=5,
    )
    ax.fill_between(
        ess_scores.index,
        ess_scores["mean"] - ess_scores["std"],
        ess_scores["mean"] + ess_scores["std"],
        alpha=0.3,
    )

    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel("Log-Likelihood Score", fontsize=12)
    ax.set_title(f"Convergence Curves (N={n_vectors} vectors, {n_steps} steps each)", fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot1 = args.output_dir / "convergence_summary.png"
    plt.savefig(plot1, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {plot1}")
    plt.close()

    # ========== Plot 2: All Trajectories Overlaid ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gradient Ascent
    for vec_id in range(min(10, n_vectors)):
        data = df_ga[df_ga["vector_id"] == vec_id]
        ax1.plot(data["step"], data["log_score"], alpha=0.5, linewidth=1)
    ax1.plot([], [], "b-", alpha=0.7, linewidth=2, label="Individual vectors")
    ax1.plot(
        ga_scores.index,
        ga_scores["mean"],
        "r-",
        linewidth=3,
        label="Mean trajectory",
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Log-Likelihood")
    ax1.set_title("Gradient Ascent (first 10 vectors)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ESS
    for vec_id in range(min(10, n_vectors)):
        data = df_ess[df_ess["vector_id"] == vec_id]
        ax2.plot(data["step"], data["log_score"], alpha=0.5, linewidth=1)
    ax2.plot([], [], "b-", alpha=0.7, linewidth=2, label="Individual vectors")
    ax2.plot(
        ess_scores.index,
        ess_scores["mean"],
        "r-",
        linewidth=3,
        label="Mean trajectory",
    )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Log-Likelihood")
    ax2.set_title("ESS (first 10 vectors)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot2 = args.output_dir / "convergence_all_vectors.png"
    plt.savefig(plot2, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {plot2}")
    plt.close()

    # ========== Plot 3: Final Score Comparison ==========
    final_ga = df_ga.groupby("vector_id")["log_score"].last()
    final_ess = df_ess.groupby("vector_id")["log_score"].last()

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = [1, 2]
    bp = ax.boxplot(
        [final_ga, final_ess],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        labels=["Gradient Ascent", "Elliptical Slice Sampling"],
    )

    # Color boxes
    for patch, color in zip(bp["boxes"], ["lightblue", "lightgreen"]):
        patch.set_facecolor(color)

    # Add individual points
    jitter1 = np.random.normal(1, 0.04, size=len(final_ga))
    jitter2 = np.random.normal(2, 0.04, size=len(final_ess))
    ax.scatter(jitter1, final_ga, alpha=0.5, s=30, color="blue")
    ax.scatter(jitter2, final_ess, alpha=0.5, s=30, color="green")

    ax.set_ylabel("Final Log-Likelihood Score", fontsize=12)
    ax.set_title("Algorithm Comparison: Final Scores", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    ga_mean = final_ga.mean()
    ess_mean = final_ess.mean()
    ax.text(
        0.5,
        0.98,
        f"GA: μ={ga_mean:.4f} | ESS: μ={ess_mean:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plot3 = args.output_dir / "algorithm_comparison.png"
    plt.savefig(plot3, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {plot3}")
    plt.close()

    print("\n" + "=" * 70)
    print("TRAJECTORY STATISTICS")
    print("=" * 70)

    initial_ga = df_ga[df_ga["step"] == 0]["log_score"].mean()
    initial_ess = df_ess[df_ess["step"] == 0]["log_score"].mean()

    print(f"\nGradient Ascent:")
    print(f"  Initial (step 0):   {initial_ga:8.4f}")
    print(f"  Final (step {n_steps-1}):     {final_ga.mean():8.4f}")
    print(f"  Improvement:        {final_ga.mean() - initial_ga:+8.4f}")
    print(f"  Std dev:            {final_ga.std():8.4f}")
    print(f"  Best score:         {final_ga.max():8.4f}")
    print(f"  Worst score:        {final_ga.min():8.4f}")

    print(f"\nElliptical Slice Sampling:")
    print(f"  Initial (step 0):   {initial_ess:8.4f}")
    print(f"  Final (step {n_steps-1}):     {final_ess.mean():8.4f}")
    print(f"  Improvement:        {final_ess.mean() - initial_ess:+8.4f}")
    print(f"  Std dev:            {final_ess.std():8.4f}")
    print(f"  Best score:         {final_ess.max():8.4f}")
    print(f"  Worst score:        {final_ess.min():8.4f}")

    winner = "ESS" if final_ess.mean() > final_ga.mean() else "Gradient Ascent"
    print(f"\n🏆 Winner: {winner}")
    print("=" * 70)


if __name__ == "__main__":
    main()
