#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
import torch
from Bio.PDB import PDBParser

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_BASE_URL = "https://files.rcsb.org/download/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download proteins and keep only VRAM-fit structures")
    parser.add_argument("--target-count", type=int, default=300)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--rows-per-page", type=int, default=500)
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    return parser


def query_ids(start: int, rows: int) -> List[str]:
    query = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": "protein"},
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    resp = requests.post(SEARCH_URL, json=query, timeout=30)
    resp.raise_for_status()
    return [item["identifier"].upper() for item in resp.json().get("result_set", [])]


def calculate_rg(pdb_path: Path) -> float | None:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", str(pdb_path))
        ca_coords = [
            res["CA"].get_coord()
            for model in structure
            for chain in model
            for res in chain
            if "CA" in res
        ]
        if not ca_coords:
            return None
        coords = np.array(ca_coords)
        center_of_mass = np.mean(coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - center_of_mass) ** 2, axis=1)))
        return float(rg)
    except Exception:
        return None


def is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main() -> None:
    args = build_parser().parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    for old in args.save_dir.glob("*.pdb"):
        old.unlink(missing_ok=True)
    args.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    if args.metadata_file.exists():
        args.metadata_file.unlink()

    import sys

    sys.path.insert(0, str(args.project_root.resolve()))
    from proteinfoundation.autoencode import ProteinAutoEncoder
    from proteinfoundation.proteinflow.proteinae import ProteinAE

    map_location = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if map_location == "cpu":
        raise RuntimeError("VRAM filtering requires CUDA. No GPU available.")

    model = ProteinAE.load_from_checkpoint(
        str(args.checkpoint),
        strict=True,
        map_location=map_location,
    )
    model = model.to(map_location)
    model.eval()
    autoencoder = ProteinAutoEncoder(model=model, trainer=None)

    accepted: Dict[str, dict] = {}
    seen_ids = set()
    page = 0

    print(f"Collecting {args.target_count} VRAM-fit proteins...")

    while len(accepted) < args.target_count and page < args.max_pages:
        ids = query_ids(start=page * args.rows_per_page, rows=args.rows_per_page)
        page += 1
        if not ids:
            break

        for pdb_id in ids:
            if pdb_id in seen_ids:
                continue
            seen_ids.add(pdb_id)
            if len(accepted) >= args.target_count:
                break

            file_path = args.save_dir / f"{pdb_id}.pdb"
            try:
                r = requests.get(f"{DOWNLOAD_BASE_URL}{pdb_id}.pdb", timeout=30)
                if r.status_code != 200:
                    continue
                file_path.write_text(r.text)

                rg_value = calculate_rg(file_path)
                if rg_value is None:
                    file_path.unlink(missing_ok=True)
                    continue

                try:
                    with torch.no_grad():
                        latent = autoencoder.encode(file_path)
                        del latent
                except Exception as exc:
                    if is_cuda_oom(exc):
                        file_path.unlink(missing_ok=True)
                        cleanup_cuda()
                        continue
                    file_path.unlink(missing_ok=True)
                    cleanup_cuda()
                    continue
                finally:
                    cleanup_cuda()

                accepted[pdb_id] = {
                    "radius_of_gyration": round(rg_value, 3),
                    "local_path": str(file_path),
                }

                if len(accepted) % 10 == 0:
                    print(f"Accepted {len(accepted)}/{args.target_count}")

            finally:
                time.sleep(args.sleep_seconds)

    if len(accepted) < args.target_count:
        raise RuntimeError(
            f"Only collected {len(accepted)} VRAM-fit proteins; target was {args.target_count}. "
            "Increase --max-pages or reduce target count."
        )

    with args.metadata_file.open("w", encoding="utf-8") as f:
        json.dump(accepted, f, indent=2)

    print(f"Done. Saved {len(accepted)} proteins to {args.save_dir}")
    print(f"Metadata: {args.metadata_file}")


if __name__ == "__main__":
    main()
