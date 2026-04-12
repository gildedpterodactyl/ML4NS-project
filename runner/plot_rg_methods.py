#!/usr/bin/env python3
"""
plot_rg_methods.py  –  Radius-of-Gyration comparison: GA vs ESS vs TESS

Extends the original 2-method Rg plot to include TESS as the third method.
Produces:
  - Violin plot:     Rg distribution per method per target Tm
  - Histogram:       Rg per method at a chosen target Tm
  - Overlay:         All three methods vs downloaded reference
  - Summary CSV:     mean / std / median Rg per method and target

Usage:
  python runner/plot_rg_methods.py \\
      --results-dir runner/results \\
      --plots-dir   runner/plots \\
      --targets 20 50 70
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── colour palette (consistent across all plots) ─────────────────────────
METHOD_COLORS = {
    "gradient_ascent": "#e07b54",   # warm orange
    "ess":             "#4f98a3",   # teal
    "tess":            "#7a39bb",   # purple  (new method)
    "downloaded":      "#b0b0b0",   # grey reference
}
METHOD_LABELS = {
    "gradient_ascent": "Gradient Ascent",
    "ess":             "ESS",
    "tess":            "TESS",
    "downloaded":      "Reference (downloaded)",
}


def load_rg(results_dir: Path, target_tm: int, method: str) -> np.ndarray | None:
    """
    Load Rg values for a (target, method) pair.
    Expected path:  results_dir/targets/target_{tm}/optimization/rg_{method}.npy
    Falls back to checking for a CSV with a 'rg' column.
    """
    base = results_dir / "targets" / f"target_{target_tm}" / "optimization"
    npy  = base / f"rg_{method}.npy"
    csv  = base / f"rg_{method}.csv"
    if npy.exists():
        return np.load(npy)
    if csv.exists():
        return pd.read_csv(csv)["rg"].values
    return None


def load_reference_rg(results_dir: Path) -> np.ndarray | None:
    for name in ("rg_downloaded.npy", "rg_reference.npy"):
        p = results_dir / name
        if p.exists():
            return np.load(p)
    for name in ("rg_downloaded.csv", "rg_reference.csv"):
        p = results_dir / name
        if p.exists():
            return pd.read_csv(p)["rg"].values
    return None


def make_violin_plot(data_dict: dict, target_tm: int, out_path: Path) -> None:
    """
    Violin plot: Rg distribution for each method at a given Tm target.
    data_dict = {method_name: np.ndarray_of_rg_values}
    """
    methods = [m for m in ["gradient_ascent", "ess", "tess"] if m in data_dict]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(
        [data_dict[m] for m in methods],
        positions=range(len(methods)),
        showmedians=True,
        showextrema=True,
    )
    for pc, m in zip(parts["bodies"], methods):
        pc.set_facecolor(METHOD_COLORS[m])
        pc.set_alpha(0.75)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("#333333")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=11)
    ax.set_ylabel("Radius of Gyration (Å)", fontsize=11)
    ax.set_title(f"Rg Distribution – Target Tm = {target_tm}°C", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_histogram(data_dict: dict, target_tm: int, out_path: Path) -> None:
    """
    Overlapping histograms (one per method) at a given Tm target.
    """
    methods = [m for m in ["gradient_ascent", "ess", "tess"] if m in data_dict]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    all_vals = np.concatenate([data_dict[m] for m in methods])
    bins = np.linspace(all_vals.min() * 0.95, all_vals.max() * 1.05, 40)

    for m in methods:
        ax.hist(
            data_dict[m],
            bins=bins,
            alpha=0.55,
            color=METHOD_COLORS[m],
            label=METHOD_LABELS[m],
            density=True,
        )
    ax.set_xlabel("Radius of Gyration (Å)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Rg Histogram – Target Tm = {target_tm}°C", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_overlay_plot(
    data_by_method: dict,
    ref_rg: np.ndarray | None,
    target_tm: int,
    out_path: Path,
) -> None:
    """
    Overlay all three generated methods + reference on a single histogram,
    for a quick visual comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    all_vals = [v for v in data_by_method.values()]
    if ref_rg is not None:
        all_vals.append(ref_rg)
    all_concat = np.concatenate(all_vals)
    bins = np.linspace(all_concat.min() * 0.95, all_concat.max() * 1.05, 45)

    for method in ["gradient_ascent", "ess", "tess"]:
        if method in data_by_method:
            ax.hist(
                data_by_method[method],
                bins=bins,
                alpha=0.45,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                density=True,
            )
    if ref_rg is not None:
        ax.hist(
            ref_rg,
            bins=bins,
            alpha=0.30,
            color=METHOD_COLORS["downloaded"],
            label=METHOD_LABELS["downloaded"],
            density=True,
            histtype="step",
            linewidth=2,
        )

    ax.set_xlabel("Radius of Gyration (Å)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Rg: All Methods vs Reference – Target Tm = {target_tm}°C", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def build_summary_csv(
    all_data: dict,   # {(target_tm, method): np.ndarray}
    out_path: Path,
) -> None:
    rows = []
    for (target_tm, method), arr in sorted(all_data.items()):
        rows.append({
            "target_tm":  target_tm,
            "method":     method,
            "n_samples":  len(arr),
            "mean_rg":    float(np.mean(arr)),
            "std_rg":     float(np.std(arr)),
            "median_rg":  float(np.median(arr)),
            "min_rg":     float(np.min(arr)),
            "max_rg":     float(np.max(arr)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  Saved summary CSV: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Rg comparison: GA vs ESS vs TESS")
    parser.add_argument("--results-dir", type=Path, default=Path("runner/results"),
                        help="Root results directory")
    parser.add_argument("--plots-dir",   type=Path, default=Path("runner/plots"),
                        help="Output directory for plots")
    parser.add_argument("--targets",     type=int,  nargs="+", default=[20, 50, 70],
                        help="Target Tm values (default: 20 50 70)")
    args = parser.parse_args()

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    ref_rg = load_reference_rg(args.results_dir)

    all_data: dict = {}
    methods = ["gradient_ascent", "ess", "tess"]

    for target_tm in args.targets:
        print(f"\n── Target Tm = {target_tm}°C ──")
        data_at_target: dict = {}

        for method in methods:
            arr = load_rg(args.results_dir, target_tm, method)
            if arr is not None and len(arr) > 0:
                data_at_target[method] = arr
                all_data[(target_tm, method)] = arr
                print(f"  {METHOD_LABELS[method]:25s}: n={len(arr):4d}  "
                      f"mean={np.mean(arr):.2f}  std={np.std(arr):.2f}")
            else:
                print(f"  {METHOD_LABELS[method]:25s}: NOT FOUND – skipping")

        if not data_at_target:
            print(f"  No data found for target {target_tm}, skipping plots.")
            continue

        make_violin_plot(
            data_at_target, target_tm,
            args.plots_dir / f"rg_violin_target_{target_tm}.png",
        )
        make_histogram(
            data_at_target, target_tm,
            args.plots_dir / f"rg_hist_target_{target_tm}.png",
        )
        make_overlay_plot(
            data_at_target, ref_rg, target_tm,
            args.plots_dir / f"rg_overlay_target_{target_tm}.png",
        )

    if all_data:
        build_summary_csv(all_data, args.plots_dir / "rg_summary_stats.csv")

    print("\nAll plots done.")


if __name__ == "__main__":
    main()
