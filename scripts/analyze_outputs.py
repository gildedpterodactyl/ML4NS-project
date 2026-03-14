#!/usr/bin/env env python3
"""
analyze_outputs.py — Collate and compare ProteinAE autoencoding results.

Reads gt.pdb / sample.pdb from each output directory, computes geometric
oracle metrics (Rg, Contact Density, H-bond proxy, Clash) and backbone
RMSD, then prints a consolidated table and optionally saves a CSV.

Usage:
    python scripts/analyze_outputs.py \
        --output_dirs output_all_baseline output_all_rg output_all_contacts output_all_hbond output_all_clash \
        --labels baseline rg contacts hbond clash \
        --save_csv results_summary.csv
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────
# Lightweight PDB parser (Cα only) — no external deps needed
# ──────────────────────────────────────────────────────────────
def parse_ca_coords(pdb_path: str) -> np.ndarray:
    """
    Parse Cα atom coordinates from a PDB file.
    Returns array of shape [n_res, 3] in Angstroms.
    """
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if (line.startswith("ATOM") or line.startswith("HETATM")):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    if len(coords) == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    return np.array(coords)


# ──────────────────────────────────────────────────────────────
# Metric functions (numpy, operating in Angstroms)
# ──────────────────────────────────────────────────────────────
def radius_of_gyration(ca: np.ndarray) -> float:
    """Radius of gyration in Angstroms."""
    com = ca.mean(axis=0)
    diff = ca - com
    rg = math.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rg


def contact_density(ca: np.ndarray, d0: float = 8.0) -> float:
    """Average number of Cα contacts per residue (hard cutoff d0 Å)."""
    n = len(ca)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = ca[:, None, :] - ca[None, :, :]  # [n, n, 3]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # [n, n]
    # Mask diagonal
    np.fill_diagonal(dist, 1e9)
    contacts = np.sum(dist < d0)
    return contacts / n


def hbond_proxy(ca: np.ndarray, d_cutoff: float = 3.5, seq_sep: int = 3) -> float:
    """H-bond proxy: fraction of Cα pairs within d_cutoff with seq sep >= seq_sep."""
    n = len(ca)
    if n < seq_sep + 1:
        return 0.0
    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, 1e9)
    count = 0
    for i in range(n):
        for j in range(i + seq_sep, n):
            if dist[i, j] < d_cutoff:
                count += 1
    return count / n


def clash_score(ca: np.ndarray, d_clash: float = 2.0, seq_sep: int = 2) -> float:
    """Clash score: average ReLU(d_clash - dist) for Cα pairs."""
    n = len(ca)
    if n < seq_sep + 1:
        return 0.0
    diff = ca[:, None, :] - ca[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    total = 0.0
    for i in range(n):
        for j in range(i + seq_sep, n):
            overlap = d_clash - dist[i, j]
            if overlap > 0:
                total += overlap
    return total / n


def rmsd(ca1: np.ndarray, ca2: np.ndarray) -> float:
    """Backbone Cα RMSD in Angstroms (no alignment, structures should already be comparable)."""
    n = min(len(ca1), len(ca2))
    diff = ca1[:n] - ca2[:n]
    return math.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def kabsch_rmsd(ca1: np.ndarray, ca2: np.ndarray) -> float:
    """Cα RMSD after optimal Kabsch alignment."""
    n = min(len(ca1), len(ca2))
    p = ca1[:n].copy()
    q = ca2[:n].copy()
    # Center
    p -= p.mean(axis=0)
    q -= q.mean(axis=0)
    # Kabsch
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    p_aligned = (R @ p.T).T
    diff = p_aligned - q
    return math.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


# ──────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────
def analyze_single(gt_path: str, sample_path: str) -> Dict[str, float]:
    """Compute all metrics for one gt/sample pair."""
    gt_ca = parse_ca_coords(gt_path)
    sample_ca = parse_ca_coords(sample_path)

    return {
        "n_res": len(gt_ca),
        "gt_rg": radius_of_gyration(gt_ca),
        "sample_rg": radius_of_gyration(sample_ca),
        "gt_contacts": contact_density(gt_ca),
        "sample_contacts": contact_density(sample_ca),
        "gt_hbond": hbond_proxy(gt_ca),
        "sample_hbond": hbond_proxy(sample_ca),
        "gt_clash": clash_score(gt_ca),
        "sample_clash": clash_score(sample_ca),
        "rmsd": kabsch_rmsd(gt_ca, sample_ca),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze ProteinAE oracle guidance outputs")
    parser.add_argument("--output_dirs", nargs="+", required=True,
                        help="Directories containing per-PDB output subdirs")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each output dir (e.g. baseline rg contacts hbond clash)")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Path to save CSV summary")
    args = parser.parse_args()

    assert len(args.output_dirs) == len(args.labels), \
        "Number of --output_dirs must match --labels"

    all_rows: List[Dict] = []
    header_printed = False

    for out_dir, label in zip(args.output_dirs, args.labels):
        if not os.path.isdir(out_dir):
            print(f"[WARN] Directory not found: {out_dir} — skipping")
            continue

        pdb_ids = sorted([
            d for d in os.listdir(out_dir)
            if os.path.isdir(os.path.join(out_dir, d))
        ])

        for pdb_id in pdb_ids:
            gt_path = os.path.join(out_dir, pdb_id, "gt.pdb")
            sample_path = os.path.join(out_dir, pdb_id, "sample.pdb")

            if not os.path.exists(gt_path) or not os.path.exists(sample_path):
                print(f"[WARN] Missing gt.pdb or sample.pdb in {out_dir}/{pdb_id}/")
                continue

            try:
                metrics = analyze_single(gt_path, sample_path)
            except Exception as e:
                print(f"[ERROR] {out_dir}/{pdb_id}: {e}")
                continue

            row = {"oracle": label, "pdb_id": pdb_id, **metrics}
            all_rows.append(row)

    if not all_rows:
        print("No results found. Check that output directories contain gt.pdb and sample.pdb files.")
        sys.exit(1)

    # ── Print table ──
    columns = [
        "oracle", "pdb_id", "n_res",
        "gt_rg", "sample_rg",
        "gt_contacts", "sample_contacts",
        "gt_hbond", "sample_hbond",
        "gt_clash", "sample_clash",
        "rmsd",
    ]

    col_widths = {c: max(len(c), 10) for c in columns}
    col_widths["oracle"] = 10
    col_widths["pdb_id"] = 8
    col_widths["n_res"] = 5

    def fmt(val, col):
        if isinstance(val, float):
            return f"{val:>{col_widths[col]}.3f}"
        elif isinstance(val, int):
            return f"{val:>{col_widths[col]}d}"
        else:
            return f"{str(val):<{col_widths[col]}}"

    # Header
    header = " | ".join(fmt(c, c) if isinstance(c, (int, float)) else f"{c:<{col_widths[c]}}" for c in columns)
    sep = "-+-".join("-" * col_widths[c] for c in columns)
    print()
    print("=" * len(sep))
    print("  ProteinAE Oracle Guidance — Results Summary")
    print("=" * len(sep))
    print(header)
    print(sep)

    prev_oracle = None
    for row in all_rows:
        if row["oracle"] != prev_oracle and prev_oracle is not None:
            print(sep)
        prev_oracle = row["oracle"]
        line = " | ".join(fmt(row[c], c) for c in columns)
        print(line)

    print(sep)

    # ── Per-oracle averages ──
    print()
    print("  Per-Oracle Averages:")
    avg_cols = ["sample_rg", "sample_contacts", "sample_hbond", "sample_clash", "rmsd"]
    avg_header = f"{'oracle':<10} | {'n':<3} | " + " | ".join(f"{c:>14}" for c in avg_cols)
    print(avg_header)
    print("-" * len(avg_header))

    oracle_labels = []
    seen = set()
    for row in all_rows:
        if row["oracle"] not in seen:
            oracle_labels.append(row["oracle"])
            seen.add(row["oracle"])

    for oracle in oracle_labels:
        oracle_rows = [r for r in all_rows if r["oracle"] == oracle]
        n = len(oracle_rows)
        avgs = []
        for col in avg_cols:
            vals = [r[col] for r in oracle_rows]
            avgs.append(sum(vals) / len(vals))
        avg_str = " | ".join(f"{v:>14.3f}" for v in avgs)
        print(f"{oracle:<10} | {n:<3} | {avg_str}")

    print()

    # ── Save CSV ──
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  CSV saved to: {args.save_csv}")
        print()


if __name__ == "__main__":
    main()
