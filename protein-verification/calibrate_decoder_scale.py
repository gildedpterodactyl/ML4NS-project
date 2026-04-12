#!/usr/bin/env python3
"""
calibrate_decoder_scale.py

Derives the empirical scale factor between the ProteinAE decoder's
normalised coordinate output and physical Angstroms, by:

    1. Loading a reference PDB (default: 3NIH)
    2. Computing native Rg from Cα atoms (ground truth in Å)
    3. Encoding the PDB → latent z  (via proteinfoundation/autoencode.py)
    4. Decoding z → reconstructed PDB
    5. Computing decoded raw Rg from the 8 Cα skeleton points
    6. scale = native_Rg / decoded_raw_Rg
    7. Writing the scale to protein-verification/decoder_scale.txt

Usage (from repo root):
    python protein-verification/calibrate_decoder_scale.py \
        --reference-pdb runner/data/3NIH.pdb \
        --ae-checkpoint /path/to/ae_r1_d8_v1.ckpt

If --skip-roundtrip is passed, the script uses the known empirical value
(native_rg=11.425 Å, decoded_rg=0.400 Å → scale=28.56) without running
the encoder/decoder, and just writes it to decoder_scale.txt.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent
SCALE_FILE  = SCRIPT_DIR / "decoder_scale.txt"

# Empirically measured from 3NIH roundtrip (ae_r1_d8_v1.ckpt)
_KNOWN_NATIVE_RG  = 11.425  # Å, from 3NIH.pdb Cα atoms
_KNOWN_DECODED_RG =  0.400  # normalised units, from decoded 3NIH skeleton


def ca_rg(pdb_path: Path) -> float:
    """Compute Cα Rg from a PDB file."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        # Fall back to all ATOM if no CA labels
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"No ATOM coords in {pdb_path}")
    arr = np.array(coords)
    center = arr.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((arr - center) ** 2, axis=1))))


def run_autoencode(input_path: Path, output_dir: Path, mode: str, checkpoint: Path) -> None:
    """Call proteinfoundation/autoencode.py via subprocess."""
    model_dir = REPO_ROOT / "model"
    env_extra = {
        "PYTHONPATH": str(model_dir),
        "AUTOENCODE_ACCELERATOR": "cpu",
    }
    import os
    env = {**os.environ, **env_extra}
    cmd = [
        sys.executable,
        str(model_dir / "proteinfoundation" / "autoencode.py"),
        "--input_pdb",  str(input_path),
        "--output_dir", str(output_dir),
        "--mode",       mode,
        "--config_path", str(model_dir / "configs"),
        "--config_name", "inference_proteinae",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(model_dir))
    if result.returncode != 0:
        raise RuntimeError(
            f"autoencode.py failed (mode={mode}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate ProteinAE decoder coordinate scale factor.")
    p.add_argument("--reference-pdb", type=Path,
                   default=REPO_ROOT / "runner" / "data" / "3NIH.pdb",
                   help="Reference PDB with known native Rg")
    p.add_argument("--ae-checkpoint",  type=Path, default=None,
                   help="Path to ae_r1_d8_v1.ckpt (required unless --skip-roundtrip)")
    p.add_argument("--skip-roundtrip", action="store_true", default=False,
                   help="Skip encode/decode; use known empirical values for 3NIH")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.skip_roundtrip:
        native_rg  = _KNOWN_NATIVE_RG
        decoded_rg = _KNOWN_DECODED_RG
        print(f"[skip-roundtrip] Using known values: native={native_rg:.4f} Å, decoded={decoded_rg:.4f}")
    else:
        # 1. Native Rg
        ref_pdb = args.reference_pdb
        if not ref_pdb.exists():
            sys.exit(f"ERROR: reference PDB not found: {ref_pdb}")
        native_rg = ca_rg(ref_pdb)
        print(f"Native Rg ({ref_pdb.name}): {native_rg:.4f} Å  ({len(open(ref_pdb).readlines())} lines)")

        if args.ae_checkpoint is None:
            sys.exit("ERROR: --ae-checkpoint required unless --skip-roundtrip")

        # 2. Encode
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path   = Path(tmp)
            enc_out    = tmp_path / "encoded"
            dec_in     = tmp_path / "latent_in"
            dec_out    = tmp_path / "decoded"

            print("Encoding reference PDB...")
            run_autoencode(ref_pdb, enc_out, "encode", args.ae_checkpoint)

            # Find latent file
            latent_files = list(enc_out.rglob("latent_repr.pt"))
            if not latent_files:
                sys.exit(f"ERROR: no latent_repr.pt found under {enc_out}")
            latent_src = latent_files[0]

            dec_in.mkdir()
            import shutil
            shutil.copy(latent_src, dec_in / "latent_repr.pt")

            print("Decoding latent...")
            run_autoencode(dec_in, dec_out, "decode", args.ae_checkpoint)

            decoded_pdbs = list(dec_out.rglob("sample.pdb"))
            if not decoded_pdbs:
                sys.exit(f"ERROR: no sample.pdb found under {dec_out}")

            decoded_rg = ca_rg(decoded_pdbs[0])
            print(f"Decoded raw Rg: {decoded_rg:.4f} (normalised units)")

    scale = native_rg / decoded_rg
    print(f"\nScale factor: {native_rg:.4f} / {decoded_rg:.4f} = {scale:.4f}")

    SCALE_FILE.write_text(str(round(scale, 6)))
    print(f"Written to: {SCALE_FILE}")
    print(f"\nAdd to gyr_pred.py: DECODER_SCALE_FACTOR = {round(scale, 2)}")


if __name__ == "__main__":
    main()
