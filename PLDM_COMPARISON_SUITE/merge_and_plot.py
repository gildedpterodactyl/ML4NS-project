import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

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


METHOD_STYLES = {
    "baseline_pldm": ("Baseline", "tab:orange"),
    "ess": ("ESS", "tab:blue"),
    "tess": ("TESS", "tab:red"),
    "transport_ess": ("Transport ESS", "tab:purple"),
    "ress": ("RESS", "tab:green"),
}


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
    p.add_argument("--test-csv", type=str, default="data/mut_data/GFP-test.csv")
    p.add_argument("--ae-ckpt", type=str, default="train_logs/GFP/epoch_1000.pt")
    p.add_argument("--dataset", type=str, default="GFP")
    p.add_argument("--baseline-results", type=str, default="baseline/results/results.csv")
    p.add_argument("--ess-results", type=str, default="ess/results/results.csv")
    p.add_argument("--tess-results", type=str, default="tess/results/results.csv")
    p.add_argument("--transport-results", type=str, default="transport/results/results.csv")
    p.add_argument("--ress-results", type=str, default="ress/results/results.csv")
    p.add_argument("--common-dir", type=str, default="common")
    p.add_argument("--results-dir", type=str, default="common/results")
    p.add_argument("--esm2-model", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--esm2-head-path", type=str, default=None)
    p.add_argument("--use-esm2", type=str2bool, default=False)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--with-structure", type=str2bool, default=False)
    p.add_argument("--wt-pdb", type=str, default="1GFL")
    p.add_argument("--umap-train-samples", type=int, default=0)
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


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def ensure_pdb(wt_pdb: str, results_dir: Path) -> Path:
    local = Path(wt_pdb)
    if local.exists():
        return local
    pdb_id = wt_pdb.upper().replace(".PDB", "")
    out = results_dir / f"{pdb_id}.pdb"
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", out)
    return out


def load_method_csv(path: Path, optional: bool = False) -> Optional[pd.DataFrame]:
    """Load a results CSV.  Returns None (rather than raising) when optional=True and missing."""
    if not path.exists():
        if optional:
            return None
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "method_name" not in df.columns:
        raise ValueError(f"Missing method_name in {path}")
    return df


def plot_stability_fitness(df: pd.DataFrame, results_dir: Path) -> None:
    if "plddt_score" not in df.columns or df["plddt_score"].isna().all():
        plt.figure(figsize=(8, 6))
        plt.text(
            0.5,
            0.5,
            "pLDDT unavailable\n(install ESMFold/OpenFold runtime to enable)",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
        plt.title("Stability-Fitness Pareto Front")
        plt.tight_layout()
        plt.savefig(results_dir / "plot_stability_fitness_pareto.png", dpi=220)
        plt.close()
        return
    pred_col = "esm2_fitness" if "esm2_fitness" in df.columns and not df["esm2_fitness"].isna().all() else "pldm_regressor_score"
    plt.figure(figsize=(8, 6))
    for m, (_, c) in METHOD_STYLES.items():
        d = df[(df["method_name"] == m) & df["plddt_score"].notna() & df[pred_col].notna()]
        plt.scatter(d["plddt_score"], d[pred_col], c=c, alpha=0.7, s=16, label=m)
    plt.axvline(70, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    plt.axvline(50, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    plt.xlabel("pLDDT score")
    plt.ylabel("Predicted fluorescence" if pred_col == "esm2_fitness" else "PRO-LDM regressor score")
    plt.title("Stability-Fitness Pareto Front")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plot_stability_fitness_pareto.png", dpi=220)
    plt.close()


def plot_perplexity_histogram(df: pd.DataFrame, results_dir: Path) -> None:
    if "perplexity" not in df.columns or df["perplexity"].isna().all():
        return
    plt.figure(figsize=(8, 6))
    bins = 25
    for m, (_, c) in METHOD_STYLES.items():
        vals = df.loc[df["method_name"] == m, "perplexity"].dropna()
        if len(vals) > 0:
            plt.hist(vals, bins=bins, alpha=0.35, density=True, color=c, label=m)
    plt.xlabel("Sequence perplexity")
    plt.ylabel("Density")
    plt.title("Perplexity Distribution Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plot_perplexity_histogram.png", dpi=220)
    plt.close()


def plot_aa_propensity_heatmap(df: pd.DataFrame, results_dir: Path) -> None:
    positions = [64, 65, 66, 67, 68, 69, 70]
    aa_vocab = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i for i, aa in enumerate(aa_vocab)}
    methods = [(m, label) for m, (label, _) in METHOD_STYLES.items()]
    matrices = []
    for method, _ in methods:
        sub = df[df["method_name"] == method]["sequence"].astype(str).tolist()
        mat = np.zeros((len(aa_vocab), len(positions)), dtype=np.float64)
        if len(sub) > 0:
            for seq in sub:
                for j, pos in enumerate(positions):
                    idx = pos - 1
                    if idx < len(seq):
                        aa = seq[idx]
                        if aa in aa_to_idx:
                            mat[aa_to_idx[aa], j] += 1.0
            mat /= float(len(sub))
        matrices.append(mat)

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    vmax = max(float(m.max()) for m in matrices) if matrices else 1.0
    for ax, (method, title), mat in zip(axes, methods, matrices):
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=max(vmax, 1e-8))
        ax.set_title(title)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([str(p) for p in positions], rotation=45)
        ax.set_yticks(range(len(aa_vocab)))
        ax.set_yticklabels(aa_vocab)
        ax.set_xlabel("Position")
    axes[0].set_ylabel("Amino acid")
    fig.suptitle("Amino Acid Propensity at Conserved GFP Positions")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Frequency")
    fig.tight_layout()
    fig.savefig(results_dir / "plot_aa_propensity_heatmap.png", dpi=220)
    plt.close(fig)


def plot_aa_propensity_heatmap_single(df: pd.DataFrame, results_dir: Path, method_name: str) -> None:
    positions = [64, 65, 66, 67, 68, 69, 70]
    aa_vocab = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i for i, aa in enumerate(aa_vocab)}
    mat = np.zeros((len(aa_vocab), len(positions)), dtype=np.float64)
    sub = df[df["method_name"] == method_name]["sequence"].astype(str).tolist()
    if len(sub) > 0:
        for seq in sub:
            for j, pos in enumerate(positions):
                idx = pos - 1
                if idx < len(seq):
                    aa = seq[idx]
                    if aa in aa_to_idx:
                        mat[aa_to_idx[aa], j] += 1.0
        mat /= float(len(sub))

    label = METHOD_STYLES.get(method_name, (method_name, "tab:gray"))[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=max(float(mat.max()), 1e-8))
    ax.set_title(label)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(p) for p in positions], rotation=45)
    ax.set_yticks(range(len(aa_vocab)))
    ax.set_yticklabels(aa_vocab)
    ax.set_xlabel("Position")
    ax.set_ylabel("Amino acid")
    fig.suptitle("Amino Acid Propensity at Conserved GFP Positions")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Frequency")
    fig.tight_layout()
    fig.savefig(results_dir / "plot_aa_propensity_heatmap.png", dpi=220)
    plt.close(fig)


def plot_umap_views(
    umap_df: pd.DataFrame,
    fitness_vals: Optional[np.ndarray],
    results_dir: Path,
    method_filter: Optional[str] = None,
) -> None:
    natural_d = umap_df[umap_df["source"] == "natural"]
    if method_filter is None:
        specs = [(m, c, 0.8, 14) for m, (_, c) in METHOD_STYLES.items()]
        title_suffix = ""
    else:
        label, color = METHOD_STYLES.get(method_filter, (method_filter, "tab:gray"))
        specs = [(method_filter, color, 0.9, 16)]
        title_suffix = f" + {label}"

    # Draw natural points first, then method points on top
    plt.figure(figsize=(12, 8))
    if fitness_vals is not None and len(fitness_vals) == len(natural_d):
        scatter = plt.scatter(
            natural_d["x"],
            natural_d["y"],
            c=fitness_vals,
            cmap="viridis",
            alpha=0.2,
            s=10,
            label="natural",
            edgecolors="none",
            zorder=1,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Fluorescence", rotation=270, labelpad=20)
    else:
        plt.scatter(
            natural_d["x"],
            natural_d["y"],
            c="lightgray",
            alpha=0.15,
            s=8,
            label="natural",
            edgecolors="none",
            zorder=1,
        )

    for src, c, a, s in specs:
        d = umap_df[umap_df["source"] == src]
        plt.scatter(d["x"], d["y"], c=c, alpha=a, s=s, label=src, edgecolors="none", zorder=2)

    plt.title(f"Latent Space UMAP (Natural = Fluorescence Heatmap{title_suffix})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_umap_latent_space.png", dpi=220)
    plt.savefig(results_dir / "plot_umap_latent_space_heatmap.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.scatter(
        natural_d["x"],
        natural_d["y"],
        c="lightgray",
        alpha=0.15,
        s=8,
        label="natural",
        edgecolors="none",
        zorder=1,
    )
    for src, c, a, s in specs:
        d = umap_df[umap_df["source"] == src]
        plt.scatter(d["x"], d["y"], c=c, alpha=a, s=s, label=src, edgecolors="none", zorder=2)
    plt.title(f"Latent Space UMAP (Natural = Gray{title_suffix})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_umap_latent_space_no_heatmap.png", dpi=220)
    plt.close()


def _hamming_lenient(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(ch1 != ch2 for ch1, ch2 in zip(a[:n], b[:n])) + abs(len(a) - len(b))


def plot_nearest_holdout_hamming_vs_fluorescence(
    generated_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    holdout_seq_col: str,
    holdout_fit_col: str,
    results_dir: Path,
) -> None:
    hold_seqs = [clean_seq(str(s)) for s in holdout_df[holdout_seq_col].tolist()]
    hold_fits = holdout_df[holdout_fit_col].astype(float).tolist()

    rows = []
    for _, row in generated_df.iterrows():
        gseq = clean_seq(str(row["sequence"]))
        dists = [_hamming_lenient(gseq, hseq) for hseq in hold_seqs]
        best_idx = int(np.argmin(dists))
        rows.append(
            {
                "method_name": row["method_name"],
                "generated_sequence": gseq,
                "nearest_holdout_sequence": hold_seqs[best_idx],
                "min_hamming_to_holdout": int(dists[best_idx]),
                "nearest_holdout_fluorescence": float(hold_fits[best_idx]),
            }
        )

    nearest_df = pd.DataFrame(rows)
    nearest_df.to_csv(results_dir / "nearest_holdout_hamming.csv", index=False)

    plt.figure(figsize=(9, 7))
    for m, (_, c) in METHOD_STYLES.items():
        d = nearest_df[nearest_df["method_name"] == m]
        if len(d) > 0:
            plt.scatter(
                d["min_hamming_to_holdout"],
                d["nearest_holdout_fluorescence"],
                c=c,
                alpha=0.7,
                s=18,
                label=m,
            )
    plt.xlabel("Minimum Hamming distance to held-out set")
    plt.ylabel("Fluorescence of nearest held-out protein")
    plt.title("Generated Proteins: Nearest Held-out Fluorescence vs Hamming Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plot_nearest_holdout_hamming_vs_fluorescence.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    proldm_root = (script_dir / args.proldm_root).resolve()
    train_csv = (proldm_root / args.train_csv).resolve()
    test_csv = (proldm_root / args.test_csv).resolve()
    ae_ckpt = (proldm_root / args.ae_ckpt).resolve()
    common_dir = (script_dir / args.common_dir).resolve()
    results_dir = (script_dir / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = (script_dir / args.baseline_results).resolve()
    ess_csv = (script_dir / args.ess_results).resolve()
    tess_csv = (script_dir / args.tess_results).resolve()
    transport_csv = (script_dir / args.transport_results).resolve()
    ress_csv = (script_dir / args.ress_results).resolve()

    # All mandatory CSVs except ress (which may not exist if mode was skipped)
    for p in [train_csv, test_csv, ae_ckpt, baseline_csv, ess_csv, tess_csv, transport_csv]:
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
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

    esm2 = ESM2Scorer(args.esm2_model, device, args.esm2_head_path) if args.use_esm2 else None

    baseline = load_method_csv(baseline_csv)
    ess = load_method_csv(ess_csv)
    tess = load_method_csv(tess_csv)
    transport = load_method_csv(transport_csv)
    ress = load_method_csv(ress_csv, optional=True)  # may be None if ress mode was skipped

    method_frames: Dict[str, pd.DataFrame] = {
        "baseline_pldm": baseline,
        "ess": ess,
        "tess": tess,
        "transport_ess": transport,
    }
    if ress is not None:
        method_frames["ress"] = ress

    combined = pd.concat(list(method_frames.values()), ignore_index=True)
    combined.to_csv(results_dir / "combined_results.csv", index=False)

    # Benchmark table
    if esm2 is not None:
        wt_score, _ = esm2.score_and_perplexity([wt_seq])
        wt_score_val = float(wt_score.detach().cpu().item())
    else:
        wt_score_val = float("nan")
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

    def _safe_mean(df, col):
        if df is None or col not in df.columns:
            return float("nan")
        return float(df[col].mean())

    table_rows = [
        {
            "Method": "Natural (WT)",
            "Target Label": "N/A",
            "ESM-2 Score": wt_score_val,
            "PRO-LDM Score": wt_reg,
            "Avg. pLDDT": wt_plddt,
            "Mean Identity": 1.0,
            "Max Identity": 1.0,
        },
        {
            "Method": "PRO-LDM (omega=20)",
            "Target Label": 8,
            "ESM-2 Score": _safe_mean(baseline, "esm2_fitness"),
            "PRO-LDM Score": _safe_mean(baseline, "pldm_regressor_score"),
            "Avg. pLDDT": _safe_mean(baseline, "plddt_score"),
            "Mean Identity": _safe_mean(baseline, "sequence_identity_wt"),
            "Max Identity": float(baseline["sequence_identity_wt"].max()) if "sequence_identity_wt" in baseline.columns else float("nan"),
        },
        {
            "Method": "ESS (TFG)",
            "Target Label": "N/A",
            "ESM-2 Score": _safe_mean(ess, "esm2_fitness"),
            "PRO-LDM Score": _safe_mean(ess, "pldm_regressor_score"),
            "Avg. pLDDT": _safe_mean(ess, "plddt_score"),
            "Mean Identity": _safe_mean(ess, "sequence_identity_wt"),
            "Max Identity": float(ess["sequence_identity_wt"].max()) if "sequence_identity_wt" in ess.columns else float("nan"),
        },
        {
            "Method": "TESS (TFG)",
            "Target Label": "N/A",
            "ESM-2 Score": _safe_mean(tess, "esm2_fitness"),
            "PRO-LDM Score": _safe_mean(tess, "pldm_regressor_score"),
            "Avg. pLDDT": _safe_mean(tess, "plddt_score"),
            "Mean Identity": _safe_mean(tess, "sequence_identity_wt"),
            "Max Identity": float(tess["sequence_identity_wt"].max()) if "sequence_identity_wt" in tess.columns else float("nan"),
        },
        {
            "Method": "Transport ESS",
            "Target Label": "N/A",
            "ESM-2 Score": _safe_mean(transport, "esm2_fitness"),
            "PRO-LDM Score": _safe_mean(transport, "pldm_regressor_score"),
            "Avg. pLDDT": _safe_mean(transport, "plddt_score"),
            "Mean Identity": _safe_mean(transport, "sequence_identity_wt"),
            "Max Identity": float(transport["sequence_identity_wt"].max()) if "sequence_identity_wt" in transport.columns else float("nan"),
        },
    ]
    if ress is not None:
        table_rows.append({
            "Method": "RESS",
            "Target Label": "N/A",
            "ESM-2 Score": _safe_mean(ress, "esm2_fitness"),
            "PRO-LDM Score": _safe_mean(ress, "pldm_regressor_score"),
            "Avg. pLDDT": _safe_mean(ress, "plddt_score"),
            "Mean Identity": _safe_mean(ress, "sequence_identity_wt"),
            "Max Identity": float(ress["sequence_identity_wt"].max()) if ress is not None and "sequence_identity_wt" in ress.columns else float("nan"),
        })

    table = pd.DataFrame(table_rows)
    table.to_csv(results_dir / "table_fitness_fidelity_benchmark.csv", index=False)
    (results_dir / "table_fitness_fidelity_benchmark.md").write_text(dataframe_to_markdown(table), encoding="utf-8")

    # UMAP
    if args.umap_train_samples is not None and args.umap_train_samples > 0:
        train_sample = train_df.sample(min(args.umap_train_samples, len(train_df)), random_state=args.seed)
    else:
        train_sample = train_df
    z_train = encode(model, [str(s) for s in train_sample[seq_col].tolist()], dims["seq_len"], device, args.batch_size)
    z_baseline = encode(model, baseline["sequence"].tolist(), dims["seq_len"], device, args.batch_size)
    z_ess = encode(model, ess["sequence"].tolist(), dims["seq_len"], device, args.batch_size)
    z_tess = encode(model, tess["sequence"].tolist(), dims["seq_len"], device, args.batch_size)
    z_transport = encode(model, transport["sequence"].tolist(), dims["seq_len"], device, args.batch_size)

    all_z_parts = [z_train, z_baseline, z_ess, z_tess, z_transport]
    sources = (
        ["natural"] * len(z_train)
        + ["baseline_pldm"] * len(z_baseline)
        + ["ess"] * len(z_ess)
        + ["tess"] * len(z_tess)
        + ["transport_ess"] * len(z_transport)
    )
    if ress is not None:
        z_ress = encode(model, ress["sequence"].tolist(), dims["seq_len"], device, args.batch_size)
        all_z_parts.append(z_ress)
        sources += ["ress"] * len(z_ress)

    all_z = torch.cat(all_z_parts, dim=0).numpy()
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=args.seed).fit_transform(all_z)

    fit_col = infer_fitness_col(train_sample)
    fitness_vals = train_sample[fit_col].values if fit_col is not None else None

    umap_df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "source": sources})
    umap_df.to_csv(results_dir / "umap_points.csv", index=False)

    plot_umap_views(umap_df=umap_df, fitness_vals=fitness_vals, results_dir=results_dir)

    # Additional quality diagnostics
    plot_stability_fitness(combined, results_dir)
    plot_perplexity_histogram(combined, results_dir)
    plot_aa_propensity_heatmap(combined, results_dir)
    holdout_seq_col = infer_seq_col(test_df)
    holdout_fit_col = infer_fitness_col(test_df)
    if holdout_fit_col is not None:
        plot_nearest_holdout_hamming_vs_fluorescence(
            generated_df=combined,
            holdout_df=test_df,
            holdout_seq_col=holdout_seq_col,
            holdout_fit_col=holdout_fit_col,
            results_dir=results_dir,
        )

    # Pareto
    if "esm2_fitness" in combined.columns and not combined["esm2_fitness"].isna().all():
        plt.figure(figsize=(9, 7))
        for m, (_, c) in METHOD_STYLES.items():
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
    fit_col_test = infer_fitness_col(test_df)
    holdout = test_df.nlargest(100, fit_col_test) if fit_col_test is not None else test_df.head(100)
    hold_dist = aa_distribution([clean_seq(str(s)) for s in holdout[seq_col].tolist()])
    kl_vals = {m: dkl(aa_distribution(df["sequence"].tolist()), hold_dist) for m, df in method_frames.items()}
    plt.figure(figsize=(8, 6))
    labels = list(kl_vals.keys())
    vals = list(kl_vals.values())
    bars = plt.bar(labels, vals, color=[METHOD_STYLES.get(l, (l, "tab:gray"))[1] for l in labels])
    for bar, lab in zip(bars, labels):
        bar.set_label(lab)
    plt.legend()
    plt.ylabel("DKL to top-100 natural test holdout")
    plt.title("KL-Divergence Bar Chart")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_kl_divergence.png", dpi=220)
    plt.close()

    # Pearson
    if "esm2_fitness" in combined.columns and not combined["esm2_fitness"].isna().all():
        plt.figure(figsize=(8, 6))
        for m, (_, c) in METHOD_STYLES.items():
            d = combined[combined["method_name"] == m]
            plt.scatter(d["esm2_fitness"], d["pldm_regressor_score"], c=c, alpha=0.6, s=14, label=m)
        plt.xlabel("ESM-2 score")
        plt.ylabel("PRO-LDM regressor score")
        plt.title("Pearson Correlation Plot")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "plot_pearson_esm2_vs_pldm.png", dpi=220)
        plt.close()

    # Per-method plot directories
    method_dirs: Dict[str, Path] = {}
    for method_name in METHOD_STYLES:
        method_dir = results_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        method_dirs[method_name] = method_dir

    for method_name, (method_label, method_color) in METHOD_STYLES.items():
        if method_name not in method_frames:
            continue  # ress may be absent
        method_df = combined[combined["method_name"] == method_name].copy()
        method_dir = method_dirs[method_name]

        plot_umap_views(umap_df=umap_df, fitness_vals=fitness_vals, results_dir=method_dir, method_filter=method_name)
        plot_stability_fitness(method_df, method_dir)
        plot_perplexity_histogram(method_df, method_dir)
        plot_aa_propensity_heatmap_single(method_df, method_dir, method_name)

        if holdout_fit_col is not None:
            plot_nearest_holdout_hamming_vs_fluorescence(
                generated_df=method_df,
                holdout_df=test_df,
                holdout_seq_col=holdout_seq_col,
                holdout_fit_col=holdout_fit_col,
                results_dir=method_dir,
            )

        if "esm2_fitness" in method_df.columns and not method_df["esm2_fitness"].isna().all():
            plt.figure(figsize=(9, 7))
            d = method_df
            plt.scatter(d["sequence_identity_wt"], d["esm2_fitness"], c=method_color, alpha=0.8, s=18, label=method_name)
            plt.xlabel("Sequence Identity to WT")
            plt.ylabel("ESM-2 Fitness")
            plt.title(f"Identity vs Fitness Pareto Front ({method_label})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(method_dir / "plot_identity_vs_fitness_pareto.png", dpi=220)
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.scatter(d["esm2_fitness"], d["pldm_regressor_score"], c=method_color, alpha=0.7, s=16, label=method_name)
            plt.xlabel("ESM-2 score")
            plt.ylabel("PRO-LDM regressor score")
            plt.title(f"Pearson Correlation Plot ({method_label})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(method_dir / "plot_pearson_esm2_vs_pldm.png", dpi=220)
            plt.close()

        method_kl = dkl(aa_distribution(method_df["sequence"].tolist()), hold_dist)
        plt.figure(figsize=(8, 6))
        plt.bar([method_name], [method_kl], color=[method_color])
        plt.ylabel("DKL to top-100 natural test holdout")
        plt.title(f"KL-Divergence Bar Chart ({method_label})")
        plt.tight_layout()
        plt.savefig(method_dir / "plot_kl_divergence.png", dpi=220)
        plt.close()

    summary = {
        "counts": {k: int(len(v)) for k, v in method_frames.items()},
        "table_path": str(results_dir / "table_fitness_fidelity_benchmark.csv"),
        "combined_csv": str(results_dir / "combined_results.csv"),
        "kl": kl_vals,
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved merged results to: {results_dir}")


if __name__ == "__main__":
    main()
