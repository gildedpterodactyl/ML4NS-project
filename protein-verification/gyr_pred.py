from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Empirically derived decoder scale factor
# ---------------------------------------------------------------------------
# The ProteinAE decoder (ae_r1_d8_v1) outputs 8 CA skeleton points in a
# normalised coordinate frame (range ≈ [-2, 2]).  The scale factor converts
# decoded Rg → physical Rg in Angstroms.
#
# Derived from 3NIH roundtrip:
#   native Rg (from PDB Cα)   = 11.425 Å
#   decoded raw Rg (8 points) ≈  0.400 Å   (measured empirically)
#   scale = 11.425 / 0.400    = 28.56
#
# Re-derive any time with:
#   python protein-verification/calibrate_decoder_scale.py
DECODER_SCALE_FACTOR: float = 28.56


def rg_from_pdb_manual(
    filename: str | Path,
    scale_factor: float = 1.0,
) -> float:
    """Compute radius of gyration from Cα atoms only.

    The autoencoder decoder outputs 8 skeleton points as ATOM/CA records in
    a normalised coordinate frame ([-2, 2]).  Pass scale_factor=DECODER_SCALE_FACTOR
    (28.56) to convert the result to physical Angstroms.

    Falls back to all ATOM records if no CA atoms are found.
    HETATM records (ligands, water, ions) are always excluded.

    Args:
        filename:     Path to PDB file.
        scale_factor: Multiply the computed Rg by this value before returning.
                      Use DECODER_SCALE_FACTOR for decoded PDBs, 1.0 for native.
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
    return rg * scale_factor


def rg_from_decoded_coords(
    coords: "np.ndarray",
    scale_factor: float = DECODER_SCALE_FACTOR,
) -> float:
    """Compute Rg directly from decoded coordinate array and rescale to Å.

    Args:
        coords:       float array of shape (N, 3) — decoded skeleton points.
        scale_factor: Multiply raw Rg to convert normalised units → Å.
                      Defaults to DECODER_SCALE_FACTOR (28.56).

    Returns:
        Rg in Angstroms.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {arr.shape}")
    center = arr.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((arr - center) ** 2, axis=1))))
    return rg * scale_factor


def _load_scale_from_file() -> Optional[float]:
    """Try to read a calibrated scale factor written by calibrate_decoder_scale.py."""
    scale_file = Path(__file__).parent / "decoder_scale.txt"
    if scale_file.exists():
        try:
            return float(scale_file.read_text().strip())
        except (ValueError, OSError):
            pass
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute radius of gyration (Cα only) for one PDB or a directory.\n"
            "Use --decoded to automatically apply the decoder coordinate rescaling "
            "(converts normalised [-2,2] decoder output to physical Angstroms)."
        )
    )
    parser.add_argument("--input_file", type=Path, default=None, help="Single PDB file")
    parser.add_argument("--input_dir",  type=Path, default=None, help="Directory with PDB files")
    parser.add_argument("--glob",       type=str,  default="*.pdb", help="Glob pattern")
    parser.add_argument("--label",      type=str,  default="sample", help="Dataset label in output CSV")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument(
        "--decoded",
        action="store_true",
        default=False,
        help=(
            "Apply decoder coordinate rescaling: multiply raw Cα Rg by "
            "DECODER_SCALE_FACTOR (28.56) to convert normalised decoder "
            "output to physical Angstroms.  Use this flag for all PDBs "
            "produced by proteinfoundation/autoencode.py decode mode."
        ),
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help=(
            "Override the scale factor explicitly (e.g. 28.56). "
            "Takes precedence over --decoded.  Useful when re-calibrating."
        ),
    )
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

    # Determine effective scale factor
    if args.scale_factor is not None:
        scale = args.scale_factor
    elif args.decoded:
        # Prefer a calibrated value from file; fall back to constant
        scale = _load_scale_from_file() or DECODER_SCALE_FACTOR
        print(f"[INFO] Applying decoder scale factor: {scale:.4f}")
    else:
        scale = 1.0

    files = collect_files(args.input_file, args.input_dir, args.glob)
    if not files:
        raise RuntimeError("No PDB files found for the given input")

    rows = []
    for pdb_file in files:
        try:
            rg = rg_from_pdb_manual(pdb_file, scale_factor=scale)
            rows.append(
                {
                    "protein_id": pdb_file.stem,
                    "path":       str(pdb_file),
                    "method":     "ca_only_scaled" if scale != 1.0 else "ca_only",
                    "dataset":    args.label,
                    "rg":         rg,
                    "scale_factor": scale,
                }
            )
        except Exception as exc:
            print(f"[WARN] Skipping {pdb_file.name}: {exc}")

    if not rows:
        raise RuntimeError("No valid Rg values computed")

    df = pd.DataFrame(rows)
    print(df.head().to_string(index=False))
    print(
        f"\nComputed Rg (Cα-only{'  ×'+str(round(scale,2)) if scale != 1.0 else ''}) "
        f"for {len(df)} proteins | dataset={args.label}"
    )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
