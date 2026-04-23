import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from pipeline_utils import (
    ESM2Scorer,
    build_hparams,
    choose_wt,
    clean_seq,
    filter_by_shape,
    hamming,
    identity,
    infer_dims_from_state,
    infer_seq_col,
    kabsch_rmsd,
    normalize_ckpt_state,
    parse_ca_coords_from_pdb,
    pdb_mean_plddt,
    seq_hash,
    seq_to_inds,
)


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    t = v.lower()
    if t in {"1", "true", "yes", "y", "t"}:
        return True
    if t in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate one method CSV into a standardized results CSV.")
    p.add_argument("--proldm-root", type=str, default="../PROLDM_OUTLIER")
    p.add_argument("--train-csv", type=str, default="data/mut_data/GFP-train.csv")
    p.add_argument("--ae-ckpt", type=str, default="train_logs/GFP/epoch_1000.pt")
    p.add_argument("--method-name", type=str, required=True)
    p.add_argument("--input-csv", type=str, required=True)
    p.add_argument("--results-csv", type=str, required=True)
    p.add_argument("--dataset", type=str, default="GFP")
    p.add_argument("--esm2-model", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--esm2-head-path", type=str, default=None)
    p.add_argument("--use-esm2", type=str2bool, default=False)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--with-structure", type=str2bool, default=False)
    p.add_argument("--wt-pdb", type=str, default="1GFL")
    p.add_argument("--structure-max-per-method", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def encode(model, seqs: List[str], seq_len: int, device: torch.device, batch_size: int) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i : i + batch_size]
            inds = torch.stack([seq_to_inds(s, seq_len) for s in chunk], dim=0).to(device)
            z = model.encode(inds).to(torch.float32)
            out.append(z.cpu())
    return torch.cat(out, dim=0)


def ensure_pdb(wt_pdb: str, results_dir: Path) -> Path:
    local = Path(wt_pdb)
    if local.exists():
        return local
    pdb_id = wt_pdb.upper().replace(".PDB", "")
    out = results_dir / f"{pdb_id}.pdb"
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", out)
    return out


def add_structure_metrics(df: pd.DataFrame, device: torch.device, wt_coords: np.ndarray, max_n: int) -> pd.DataFrame:
    try:
        import esm
        from tmtools import tm_align
        model = esm.pretrained.esmfold_v1().eval().to(device)
        model.set_chunk_size(128)
    except ModuleNotFoundError:
        df["plddt_score"] = np.nan
        df["tm_score"] = np.nan
        df["rmsd"] = np.nan
        return df
    except Exception:
        df["plddt_score"] = np.nan
        df["tm_score"] = np.nan
        df["rmsd"] = np.nan
        return df

    df = df.copy()
    df["plddt_score"] = np.nan
    df["tm_score"] = np.nan
    df["rmsd"] = np.nan

    order = df.sort_values("esm2_fitness", ascending=False).head(max_n).index
    with torch.no_grad():
        for idx in order:
            seq = str(df.loc[idx, "sequence"])
            try:
                pdb_txt = model.infer_pdb(seq)
                plddt = pdb_mean_plddt(pdb_txt)
                tmp = Path("/tmp") / f"cmp_{idx}.pdb"
                tmp.write_text(pdb_txt)
                pred_coords = parse_ca_coords_from_pdb(tmp)
                n = min(len(pred_coords), len(wt_coords), len(seq))
                if n > 0:
                    align = tm_align(pred_coords[:n], wt_coords[:n], seq[:n], seq[:n])
                    tm_score = float(align.tm_norm_chain1)
                    rmsd = kabsch_rmsd(pred_coords[:n], wt_coords[:n])
                else:
                    tm_score = np.nan
                    rmsd = np.nan
            except Exception:
                plddt = np.nan
                tm_score = np.nan
                rmsd = np.nan
            df.loc[idx, "plddt_score"] = plddt
            df.loc[idx, "tm_score"] = tm_score
            df.loc[idx, "rmsd"] = rmsd

    return df


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    proldm_root = (script_dir / args.proldm_root).resolve()
    train_csv = (proldm_root / args.train_csv).resolve()
    ae_ckpt = (proldm_root / args.ae_ckpt).resolve()
    input_csv = (script_dir / args.input_csv).resolve()
    results_csv = (script_dir / args.results_csv).resolve()
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    for p in [train_csv, ae_ckpt, input_csv]:
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    train_df = pd.read_csv(train_csv)
    wt_seq = choose_wt(train_df)

    sys.path.insert(0, str(proldm_root))
    from model.JTAE.models_condif_1d import jtae

    ckpt_raw = torch.load(ae_ckpt, map_location=device)
    state = normalize_ckpt_state(ckpt_raw)
    dims = infer_dims_from_state(state)
    hparams = build_hparams(args.dataset, dims, device)
    model = jtae(hparams).to(device)
    model.load_state_dict(filter_by_shape(state, model.state_dict()), strict=False)
    model.eval()

    raw = pd.read_csv(input_csv)
    seq_col = infer_seq_col(raw)
    seqs = [clean_seq(str(s)) for s in raw[seq_col].tolist()]

    z = encode(model, seqs, dims["seq_len"], device, args.batch_size).to(device)
    with torch.no_grad():
        pldm = model.regressor_module(z).squeeze(-1).detach().cpu().numpy()

    esm2 = ESM2Scorer(args.esm2_model, device, args.esm2_head_path) if args.use_esm2 else None
    if esm2 is not None:
        esm_scores = []
        perplexities = []
        with torch.no_grad():
            for i in range(0, len(seqs), args.batch_size):
                chunk = seqs[i : i + args.batch_size]
                s, p = esm2.score_and_perplexity(chunk)
                esm_scores.extend(s.detach().cpu().tolist())
                perplexities.extend(p.detach().cpu().tolist())
    else:
        esm_scores = [np.nan] * len(seqs)
        perplexities = [np.nan] * len(seqs)

    out = pd.DataFrame(
        {
            "sequence": seqs,
            "id": [seq_hash(s) for s in seqs],
            "method_name": args.method_name,
            "sequence_identity_wt": [identity(s, wt_seq) for s in seqs],
            "hamming_dist_wt": [hamming(s, wt_seq) for s in seqs],
            "esm2_fitness": esm_scores,
            "pldm_regressor_score": pldm,
            "perplexity": perplexities,
        }
    )

    if args.with_structure:
        try:
            wt_pdb = ensure_pdb(args.wt_pdb, results_csv.parent)
            wt_coords = parse_ca_coords_from_pdb(wt_pdb)
            out = add_structure_metrics(out, device, wt_coords, args.structure_max_per_method)
        except ModuleNotFoundError:
            out["plddt_score"] = np.nan
            out["tm_score"] = np.nan
            out["rmsd"] = np.nan
    else:
        out["plddt_score"] = np.nan
        out["tm_score"] = np.nan
        out["rmsd"] = np.nan

    out.to_csv(results_csv, index=False)
    print(f"Saved standardized results to: {results_csv}")


if __name__ == "__main__":
    main()
