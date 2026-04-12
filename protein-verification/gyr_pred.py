from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def rg_from_pdb_manual(filename: str | Path) -> float:
    """Compute radius of gyration from Cα atoms only.

    The autoencoder decoder outputs 8 skeleton points written as ATOM records.
    Using all-atom Rg on 8 points gives ~1 Å regardless of structure — meaningless.
    Filtering to CA gives the correct per-residue Rg in the model's Å coordinate frame.

    Falls back to all ATOM records if no CA atoms are found (safety net for
    non-standard or coarse-grained PDB files that label points differently).
    HETATM records (ligands, water, ions) are always excluded.
    """
    ca_coords: List[List[float]] = []
    all_atom_coords: List[List[float]] = []

    with open(filename, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            all_atom_coords.append([x, y, z])
            if atom_name == "CA":
                ca_coords.append([x, y, z])

    coords = ca_coords if ca_coords else all_atom_coords

    if not coords:
        raise ValueError(f"No ATOM coordinates found in {filename}")

    arr = np.array(coords, dtype=np.float64)
    center = arr.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((arr - center) ** 2, axis=1))))
    return rg


def rg_from_decoded_coords(coords: "np.ndarray") -> float:
    """Compute Rg directly from decoded coordinate array.

    Args:
        coords: float array of shape (N, 3) in Angstroms — the N skeleton
                points output by model.decode().  No PDB roundtrip needed.

    Returns:
        Rg in Angstroms.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {arr.shape}")
    center = arr.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((arr - center) ** 2, axis=1))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute radius of gyration (Cα only) for one PDB or a directory."
    )
    parser.add_argument("--input_file", type=Path, default=None, help="Single PDB file")
    parser.add_argument("--input_dir", type=Path, default=None, help="Directory with PDB files")
    parser.add_argument("--glob", type=str, default="*.pdb", help="Glob pattern for PDB files")
    parser.add_argument("--label", type=str, default="sample", help="Dataset label in output CSV")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional CSV output path")
    return parser.parse_args()


def collect_files(
    input_file: Optional[Path], input_dir: Optional[Path], glob_pattern: str
) -> List[Path]:
    if input_file is not None:
        return [input_file]
    if input_dir is None:
        raise ValueError("Provide either --input_file or --input_dir")
    return sorted(input_dir.glob(glob_pattern))


def main() -> None:
    args = parse_args()
    files = collect_files(args.input_file, args.input_dir, args.glob)
    if not files:
        raise RuntimeError("No PDB files found for the given input")

    rows = []
    for pdb_file in files:
        try:
            rg = rg_from_pdb_manual(pdb_file)
            rows.append(
                {
                    "protein_id": pdb_file.stem,
                    "path": str(pdb_file),
                    "method": "ca_only",
                    "dataset": args.label,
                    "rg": rg,
                }
            )
        except Exception as exc:
            print(f"[WARN] Skipping {pdb_file.name}: {exc}")

    if not rows:
        raise RuntimeError("No valid Rg values computed")

    df = pd.DataFrame(rows)
    print(df.head().to_string(index=False))
    print(f"\nComputed Rg (Cα-only) for {len(df)} proteins | dataset={args.label}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
