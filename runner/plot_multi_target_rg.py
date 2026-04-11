#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot multi-target Rg comparisons")
    parser.add_argument("--targets-root", type=Path, required=True)
    parser.add_argument("--downloaded-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", type=str, default="20,50,70")
    return parser


def load_rg(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "rg" not in df.columns:
        raise ValueError(f"Missing rg column: {path}")
    return df["rg"].astype(float)


def save_compare_hist(gradient: pd.Series, slice_sampling: pd.Series, target: float, out: Path) -> None:
    all_vals = np.concatenate([gradient.to_numpy(), slice_sampling.to_numpy()])
    bins = np.linspace(float(all_vals.min()), float(all_vals.max()), 30)

    plt.figure(figsize=(10, 6))
    plt.hist(gradient.to_numpy(), bins=bins, alpha=0.5, label=f"gradient_ascent (n={len(gradient)})")
    plt.hist(slice_sampling.to_numpy(), bins=bins, alpha=0.5, label=f"slice_sampling (n={len(slice_sampling)})")
    plt.axvline(target, linestyle="--", color="black", linewidth=1.5, label=f"target={target}")
    plt.xlabel("Radius of gyration (Å)")
    plt.ylabel("Count")
    plt.title(f"Generated Rg histogram: target {target:.1f} Å")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def save_compare_violin(gradient: pd.Series, slice_sampling: pd.Series, target: float, out: Path) -> None:
    plt.figure(figsize=(9, 6))
    parts = plt.violinplot(
        [gradient.to_numpy(), slice_sampling.to_numpy()],
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for body in parts["bodies"]:
        body.set_alpha(0.85)

    plt.xticks([1, 2], [f"gradient_ascent\n(n={len(gradient)})", f"slice_sampling\n(n={len(slice_sampling)})"])
    plt.axhline(target, linestyle="--", color="black", linewidth=1.5, label=f"target={target}")
    plt.ylabel("Radius of gyration (Å)")
    plt.title(f"Generated Rg violin: target {target:.1f} Å")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def save_downloaded_hist(downloaded: pd.Series, out: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(downloaded.to_numpy(), bins=30, alpha=0.8)
    plt.xlabel("Radius of gyration (Å)")
    plt.ylabel("Count")
    plt.title(f"Downloaded proteins Rg histogram (n={len(downloaded)})")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]

    downloaded = load_rg(args.downloaded_csv)
    save_downloaded_hist(downloaded, args.output_dir / "rg_hist_downloaded.png")

    summary_rows = []
    for t in targets:
        tag = str(int(t)) if float(t).is_integer() else str(t).replace(".", "p")
        target_dir = args.targets_root / f"target_{tag}" / "validation"

        grad = load_rg(target_dir / "rg_gradient_ascent.csv")
        ess = load_rg(target_dir / "rg_slice_sampling.csv")

        save_compare_hist(grad, ess, t, args.output_dir / f"rg_hist_target_{tag}.png")
        save_compare_violin(grad, ess, t, args.output_dir / f"rg_violin_target_{tag}.png")

        summary_rows.append(
            {
                "target": t,
                "gradient_mean": float(grad.mean()),
                "slice_mean": float(ess.mean()),
                "gradient_median": float(grad.median()),
                "slice_median": float(ess.median()),
                "gradient_n": int(len(grad)),
                "slice_n": int(len(ess)),
            }
        )

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "rg_target_summary.csv", index=False)

    print("Saved plot files:")
    for p in sorted(args.output_dir.glob("*.png")):
        print(f" - {p}")
    print(f"Saved summary: {args.output_dir / 'rg_target_summary.csv'}")


if __name__ == "__main__":
    main()
