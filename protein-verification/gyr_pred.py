from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def rg_from_pdb_manual(filename: str | Path) -> float:
    coords = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

    if not coords:
        raise ValueError(f"No atom coordinates found in {filename}")

    arr = np.array(coords)
    center = np.mean(arr, axis=0)
    rg = np.sqrt(np.mean(np.sum((arr - center) ** 2, axis=1)))
    return float(rg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute radius of gyration for one file or a directory.")
    parser.add_argument("--input_file", type=Path, default=None, help="Single PDB file path")
    parser.add_argument("--input_dir", type=Path, default=None, help="Directory with PDB files")
    parser.add_argument("--glob", type=str, default="*.pdb", help="Glob pattern for PDB files")
    parser.add_argument("--label", type=str, default="sample", help="Dataset label in output CSV")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional CSV output path")
    return parser.parse_args()


def collect_files(input_file: Path | None, input_dir: Path | None, glob_pattern: str) -> List[Path]:
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
                    "method": "manual_all_atom",
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
    print(f"\nComputed Rg for {len(df)} proteins | dataset={args.label}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()