import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap

from pipeline_utils import (
    build_hparams,
    choose_wt,
    clean_seq,
    filter_by_shape,
    infer_dims_from_state,
    infer_fitness_col,
    infer_seq_col,
    kabsch_rmsd,
    normalize_ckpt_state,
    parse_ca_coords_from_pdb,
    pdb_mean_plddt,
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
    p = argparse.ArgumentParser(description="Merge method CSVs and create common plots/tables.")
    p.add_argument("--proldm-root", type=str, default="../PROLDM_OUTLIER")
    p.add_argument("--train-csv", type=str, default="data/mut_data/GFP-train.csv")
    p.add_argument("--ae-ckpt", type=str, default="train_logs/GFP/epoch_1000.pt")
    p.add_argument("--dataset", type=str, default="GFP")
    p.add_argument("--baseline-results", type=str, default="baseline/results/results.csv")
    p.add_argument("--ess-results", type=str, default="ess/results/results.csv")
    p.add_argument("--tess-results", type=str, default="tess/results/results.csv")
    p.add_argument("--common-dir", type=str, default="common")
    p.add_argument("--results-dir", type=str, default="common/results")
    p.add_argument("--esm2-model", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--esm2-head-path", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--with-structure", type=str2bool, default=False)
    p.add_argument("--wt-pdb", type=str, default="1GFL")
    p.add_argument("--umap-train-samples", type=int, default=5000)
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


def aa_distribution(seqs: List[str]) -> np.ndarray:
    vocab = list("ILVFMCAGPTSYWQNHEDKRX")
    idx = {a: i for i, a in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=np.float64)
    total = 0
    for s in seqs:
        for ch in s:
            counts[idx[ch if ch in idx else "X"]] += 1
            total += 1
    return (counts + 1e-8) / (total + 1e-8 * len(vocab))


def dkl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


def ensure_pdb(wt_pdb: str, results_dir: Path) -> Path:
    local = Path(wt_pdb)
    if local.exists():
        return local
    pdb_id = wt_pdb.upper().replace(".PDB", "")
    out = results_dir / f"{pdb_id}.pdb"
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", out)
    return out


def load_method_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method_name" not in df.columns:
        raise ValueError(f"Missing method_name in {path}")
    return df


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    proldm_root = (script_dir / args.proldm_root).resolve()
    train_csv = (proldm_root / args.train_csv).resolve()
    ae_ckpt = (proldm_root / args.ae_ckpt).resolve()
    common_dir = (script_dir / args.common_dir).resolve()
    results_dir = (script_dir / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = (script_dir / args.baseline_results).resolve()
    ess_csv = (script_dir / args.ess_results).resolve()
    tess_csv = (script_dir / args.tess_results).resolve()
    for p in [train_csv, ae_ckpt, baseline_csv, ess_csv, tess_csv]:
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    train_df = pd.read_csv(train_csv)
    wt_seq = choose_wt(train_df)
    seq_col = infer_seq_col(train_df)

    sys.path.insert(0, str(proldm_root))
    from model.JTAE.models_condif_1d import jtae
    from pipeline_utils import ESM2Scorer

    ckpt_raw = torch.load(ae_ckpt, map_location=device)
    state = normalize_ckpt_state(ckpt_raw)
    dims = infer_dims_from_state(state)
    hparams = build_hparams(args.dataset, dims, device)
    model = jtae(hparams).to(device)
    model.load_state_dict(filter_by_shape(state, model.state_dict()), strict=False)
    model.eval()

    esm2 = ESM2Scorer(args.esm2_model, device, args.esm2_head_path)

    baseline = load_method_csv(baseline_csv)
    ess = load_method_csv(ess_csv)
    tess = load_method_csv(tess_csv)

    combined = pd.concat([baseline, ess, tess], ignore_index=True)
    combined.to_csv(results_dir / "combined_results.csv", index=False)

    # Benchmark table
    wt_score, _ = esm2.score_and_perplexity([wt_seq])
    with torch.no_grad():
        wt_z = encode(model, [wt_seq], dims["seq_len"], device, args.batch_size).to(device)
        wt_reg = model.regressor_module(wt_z).squeeze(-1).detach().cpu().item()
    wt_plddt = float("nan")
    if args.with_structure:
        try:
            import esm

            wt_pdb = ensure_pdb(args.wt_pdb, results_dir)
            fold_model = esm.pretrained.esmfold_v1().eval().to(device)
            fold_model.set_chunk_size(128)
            wt_plddt = pdb_mean_plddt(fold_model.infer_pdb(wt_seq))
        except ModuleNotFoundError:
            wt_plddt = float("nan")
        except Exception:
            wt_plddt = float("nan")

    table = pd.DataFrame(
        [
            {
                "Method": "Natural (WT)",
                "Target Label": "N/A",
                "ESM-2 Score": float(wt_score.detach().cpu().item()),
                "PRO-LDM Score": wt_reg,
                "Avg. pLDDT": wt_plddt,
                "Mean Identity": 1.0,
                "Max Identity": 1.0,
            },
            {
                "Method": "PRO-LDM (ω=20)",
                "Target Label": 8,
                "ESM-2 Score": float(baseline["esm2_fitness"].mean()),
                "PRO-LDM Score": float(baseline["pldm_regressor_score"].mean()),
                "Avg. pLDDT": float(baseline["plddt_score"].mean()),
                "Mean Identity": float(baseline["sequence_identity_wt"].mean()),
                "Max Identity": float(baseline["sequence_identity_wt"].max()),
            },
            {
                "Method": "ESS (TFG)",
                "Target Label": "N/A",
                "ESM-2 Score": float(ess["esm2_fitness"].mean()),
                "PRO-LDM Score": float(ess["pldm_regressor_score"].mean()),
                "Avg. pLDDT": float(ess["plddt_score"].mean()),
                "Mean Identity": float(ess["sequence_identity_wt"].mean()),
                "Max Identity": float(ess["sequence_identity_wt"].max()),
            },
            {
                "Method": "TESS (TFG)",
                "Target Label": "N/A",
                "ESM-2 Score": float(tess["esm2_fitness"].mean()),
                "PRO-LDM Score": float(tess["pldm_regressor_score"].mean()),
                "Avg. pLDDT": float(tess["plddt_score"].mean()),
                "Mean Identity": float(tess["sequence_identity_wt"].mean()),
                "Max Identity": float(tess["sequence_identity_wt"].max()),
            },
        ]
    )
    table.to_csv(results_dir / "table_fitness_fidelity_benchmark.csv", index=False)
    (results_dir / "table_fitness_fidelity_benchmark.md").write_text(table.to_markdown(index=False), encoding="utf-8")

    # UMAP
    train_sample = train_df.sample(min(args.umap_train_samples, len(train_df)), random_state=args.seed)
    z_train = encode(model, [str(s) for s in train_sample[seq_col].tolist()], dims["seq_len"], device, args.batch_size)
    z_baseline = encode(model, baseline["sequence"].tolist(), dims["seq_len"], device, args.batch_size)
    z_ess = encode(model, ess["sequence"].tolist(), dims["seq_len"], device, args.batch_size)
    z_tess = encode(model, tess["sequence"].tolist(), dims["seq_len"], device, args.batch_size)
    all_z = torch.cat([z_train, z_baseline, z_ess, z_tess], dim=0).numpy()
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=args.seed).fit_transform(all_z)
    sources = ["natural"] * len(z_train) + ["baseline_pldm"] * len(z_baseline) + ["ess"] * len(z_ess) + ["tess"] * len(z_tess)
    umap_df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "source": sources})
    umap_df.to_csv(results_dir / "umap_points.csv", index=False)

    plt.figure(figsize=(10, 8))
    legend_specs = [("natural", "lightgray", 0.3, 8), ("baseline_pldm", "tab:blue", 0.8, 14), ("ess", "tab:orange", 0.8, 14), ("tess", "tab:green", 0.8, 14)]
    for src, c, a, s in legend_specs:
        d = umap_df[umap_df["source"] == src]
        plt.scatter(d["x"], d["y"], c=c, alpha=a, s=s, label=src)
    plt.title("Latent Space UMAP")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plot_umap_latent_space.png", dpi=220)
    plt.close()

    # Pareto
    plt.figure(figsize=(9, 7))
    for m, c in [("baseline_pldm", "tab:blue"), ("ess", "tab:orange"), ("tess", "tab:green")]:
        d = combined[combined["method_name"] == m]
        plt.scatter(d["sequence_identity_wt"], d["esm2_fitness"], c=c, alpha=0.7, s=16, label=m)
    plt.xlabel("Sequence Identity to WT")
    plt.ylabel("ESM-2 Fitness")
    plt.title("Identity vs Fitness Pareto Front")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plot_identity_vs_fitness_pareto.png", dpi=220)
    plt.close()

    # KL
    fit_col = infer_fitness_col(train_df)
    holdout = train_df.nlargest(100, fit_col) if fit_col is not None else train_df.head(100)
    hold_dist = aa_distribution([clean_seq(str(s)) for s in holdout[seq_col].tolist()])
    kl_vals = {
        "baseline_pldm": dkl(aa_distribution(baseline["sequence"].tolist()), hold_dist),
        "ess": dkl(aa_distribution(ess["sequence"].tolist()), hold_dist),
        "tess": dkl(aa_distribution(tess["sequence"].tolist()), hold_dist),
    }
    plt.figure(figsize=(8, 6))
    labels = list(kl_vals.keys())
    vals = list(kl_vals.values())
    bars = plt.bar(labels, vals, color=["tab:blue", "tab:orange", "tab:green"])
    for bar, lab in zip(bars, labels):
        bar.set_label(lab)
    plt.legend()
    plt.ylabel("DKL to top-100 natural holdout")
    plt.title("KL-Divergence Bar Chart")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_kl_divergence.png", dpi=220)
    plt.close()

    # Pearson
    plt.figure(figsize=(8, 6))
    for m, c in [("baseline_pldm", "tab:blue"), ("ess", "tab:orange"), ("tess", "tab:green")]:
        d = combined[combined["method_name"] == m]
        plt.scatter(d["esm2_fitness"], d["pldm_regressor_score"], c=c, alpha=0.6, s=14, label=m)
    plt.xlabel("ESM-2 score")
    plt.ylabel("PRO-LDM regressor score")
    plt.title("Pearson Correlation Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plot_pearson_esm2_vs_pldm.png", dpi=220)
    plt.close()

    summary = {
        "counts": {"baseline_pldm": int(len(baseline)), "ess": int(len(ess)), "tess": int(len(tess))},
        "table_path": str(results_dir / "table_fitness_fidelity_benchmark.csv"),
        "combined_csv": str(results_dir / "combined_results.csv"),
        "kl": kl_vals,
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved merged results to: {results_dir}")


if __name__ == "__main__":
    main()
