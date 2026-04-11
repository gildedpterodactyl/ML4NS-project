#!/usr/bin/env python3
"""Plot radius-of-gyration comparisons for feed-forward and generated proteins."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Rg comparisons across methods")
    parser.add_argument(
        "--validation-dir",
        type=Path,
        required=True,
        help="Directory containing rg_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where plots should be written",
    )
    parser.add_argument(
        "--target-value",
        type=float,
        default=None,
        help="Optional target Rg value for target-distance plots",
    )
    return parser


def load_series(path: Path, label: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing Rg CSV for {label}: {path}")
    df = pd.read_csv(path)
    if "rg" not in df.columns:
        raise ValueError(f"Expected an 'rg' column in {path}")
    return df["rg"].astype(float)


def save_line_plot(series_map: dict[str, pd.Series], output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    for label, series in series_map.items():
        x = np.arange(1, len(series) + 1)
        plt.plot(x, series.to_numpy(), marker="o", markersize=3, linewidth=1.2, label=f"{label} (n={len(series)})")
    plt.xlabel("Sample index")
    plt.ylabel("Radius of gyration (Å)")
    plt.title("Rg of all samples by method")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_distribution_plot(series_map: dict[str, pd.Series], output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    max_val = max(float(series.max()) for series in series_map.values())
    min_val = min(float(series.min()) for series in series_map.values())
    bins = np.linspace(min_val, max_val, 30)
    for label, series in series_map.items():
        plt.hist(series.to_numpy(), bins=bins, alpha=0.45, density=True, label=f"{label} (n={len(series)})")
    plt.xlabel("Radius of gyration (Å)")
    plt.ylabel("Density")
    plt.title("Rg distribution by method")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_violin_plot(series_map: dict[str, pd.Series], output_path: Path) -> None:
    labels = list(series_map.keys())
    data = [series_map[label].to_numpy() for label in labels]
    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#8ecae6")
        body.set_edgecolor("#3a7ca5")
        body.set_alpha(0.8)
    plt.xticks(range(1, len(labels) + 1), [f"{label}\n(n={len(series_map[label])})" for label in labels])
    plt.ylabel("Radius of gyration (Å)")
    plt.title("Rg violin plot by method")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_generated_only_violin(series_map: dict[str, pd.Series], output_path: Path) -> None:
    labels = ["gradient_ascent", "slice_sampling"]
    data = [series_map[label].to_numpy() for label in labels]

    plt.figure(figsize=(9, 6))
    parts = plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#bde0fe")
        body.set_edgecolor("#1d3557")
        body.set_alpha(0.85)

    plt.xticks(range(1, len(labels) + 1), [f"{label}\n(n={len(series_map[label])})" for label in labels])
    plt.ylabel("Radius of gyration (Å)")
    plt.title("Generated-only Rg violin plot")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_target_distance_hist(series_map: dict[str, pd.Series], target_value: float, output_path: Path) -> pd.DataFrame:
    gradient_dist = np.abs(series_map["gradient_ascent"].to_numpy() - target_value)
    slice_dist = np.abs(series_map["slice_sampling"].to_numpy() - target_value)

    max_dist = float(max(gradient_dist.max(initial=0.0), slice_dist.max(initial=0.0)))
    bins = np.linspace(0.0, max_dist if max_dist > 0 else 1.0, 30)

    plt.figure(figsize=(10, 6))
    plt.hist(gradient_dist, bins=bins, alpha=0.5, label=f"gradient_ascent (n={len(gradient_dist)})")
    plt.hist(slice_dist, bins=bins, alpha=0.5, label=f"slice_sampling (n={len(slice_dist)})")
    plt.xlabel("Absolute distance to target |Rg - target| (Å)")
    plt.ylabel("Count")
    plt.title(f"How close generated proteins are to target Rg = {target_value:.2f} Å")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    thresholds = [0.5, 1.0, 2.0, 3.0]
    rows = []
    for method, dist in [("gradient_ascent", gradient_dist), ("slice_sampling", slice_dist)]:
        row = {
            "method": method,
            "n_samples": int(len(dist)),
            "mean_abs_distance": float(np.mean(dist)) if len(dist) else np.nan,
            "median_abs_distance": float(np.median(dist)) if len(dist) else np.nan,
        }
        for t in thresholds:
            row[f"within_{t:.1f}A_count"] = int(np.sum(dist <= t))
            row[f"within_{t:.1f}A_fraction"] = float(np.mean(dist <= t)) if len(dist) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def save_summary_stats(series_map: dict[str, pd.Series], output_path: Path) -> None:
    rows = []
    for label, series in series_map.items():
        rows.append(
            {
                "method": label,
                "count": int(len(series)),
                "mean_rg": float(series.mean()),
                "std_rg": float(series.std()),
                "min_rg": float(series.min()),
                "max_rg": float(series.max()),
                "median_rg": float(series.median()),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    series_map = {
        "initial_feedforward": load_series(args.validation_dir / "rg_initial_feedforward.csv", "initial feed-forward data"),
        "gradient_ascent": load_series(args.validation_dir / "rg_gradient_ascent.csv", "gradient ascent"),
        "slice_sampling": load_series(args.validation_dir / "rg_slice_sampling.csv", "slice sampling"),
    }

    save_line_plot(series_map, args.output_dir / "rg_samples_by_method.png")
    save_distribution_plot(series_map, args.output_dir / "rg_distribution_overlay.png")
    save_violin_plot(series_map, args.output_dir / "rg_violin.png")
    save_generated_only_violin(series_map, args.output_dir / "rg_violin_generated_only.png")
    save_summary_stats(series_map, args.output_dir / "rg_summary_stats.csv")

    if args.target_value is not None:
        proximity_df = save_target_distance_hist(
            series_map,
            target_value=args.target_value,
            output_path=args.output_dir / "rg_target_distance_hist.png",
        )
        proximity_df.to_csv(args.output_dir / "rg_target_proximity_summary.csv", index=False)

    print("Saved plots:")
    for path in sorted(args.output_dir.glob("*.png")):
        print(f" - {path}")
    print(f"Saved summary stats: {args.output_dir / 'rg_summary_stats.csv'}")
    if args.target_value is not None:
        print(f"Saved target proximity summary: {args.output_dir / 'rg_target_proximity_summary.csv'}")


if __name__ == "__main__":
    main()
