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
    ESM2Scorer,
    build_hparams,
    choose_wt,
    clean_seq,
    filter_by_shape,
    hamming,
    identity,
    infer_dims_from_state,
    infer_fitness_col,
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
    p = argparse.ArgumentParser(description="Compute metrics and plots for baseline/ESS/TESS comparison.")
    p.add_argument("--proldm-root", type=str, default="../PROLDM_OUTLIER")
    p.add_argument("--train-csv", type=str, default="data/mut_data/GFP-train.csv")
    p.add_argument("--test-csv", type=str, default="data/mut_data/GFP-test.csv")
    p.add_argument("--ae-ckpt", type=str, default="train_logs/GFP/epoch_1000.pt")
    p.add_argument("--dataset", type=str, default="GFP")
    p.add_argument("--raw-dir", type=str, default="outputs")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--esm2-model", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--esm2-head-path", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--with-structure", type=str2bool, default=False)
    p.add_argument("--wt-pdb", type=str, default="1GFL")
    p.add_argument("--structure-max-per-method", type=int, default=100)
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


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def summarize_method(df: pd.DataFrame, method_name: str, target_label) -> Dict[str, object]:
    return {
        "Method": method_name,
        "Target Label": target_label,
        "ESM-2 Score": float(df["esm2_fitness"].mean()),
        "PRO-LDM Score": float(df["pldm_regressor_score"].mean()),
        "Avg. pLDDT": float(df["plddt_score"].mean()),
        "Mean Identity": float(df["sequence_identity_wt"].mean()),
        "Max Identity": float(df["sequence_identity_wt"].max()),
    }


def write_comparative_table(
    results_dir: Path,
    wt_seq: str,
    wt_identity_df: pd.DataFrame,
    baseline: pd.DataFrame,
    ess: pd.DataFrame,
    tess: pd.DataFrame,
    wt_plddt: float,
    wt_esm2_score: float,
    wt_reg_score: float,
) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            {
                "Method": "Natural (WT)",
                "Target Label": "N/A",
                "ESM-2 Score": wt_esm2_score,
                "PRO-LDM Score": wt_reg_score,
                "Avg. pLDDT": wt_plddt,
                "Mean Identity": 1.0,
                "Max Identity": 1.0,
            },
            summarize_method(baseline, "PRO-LDM (ω=20)", 8),
            summarize_method(ess, "ESS (TFG)", "N/A"),
            summarize_method(tess, "TESS (TFG)", "N/A"),
        ]
    )
    table.to_csv(results_dir / "table_fitness_fidelity_benchmark.csv", index=False)
    (results_dir / "table_fitness_fidelity_benchmark.md").write_text(dataframe_to_markdown(table), encoding="utf-8")


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
    for m, c in [("baseline_pldm", "tab:orange"), ("ess", "tab:blue"), ("tess", "tab:red")]:
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
    for m, c in [("baseline_pldm", "tab:orange"), ("ess", "tab:blue"), ("tess", "tab:red")]:
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
    methods = [("baseline_pldm", "Baseline"), ("ess", "ESS"), ("tess", "TESS")]
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
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
    for m, c in [("baseline_pldm", "tab:orange"), ("ess", "tab:blue"), ("tess", "tab:red")]:
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


def ensure_pdb(wt_pdb: str, results_dir: Path) -> Path:
    local = Path(wt_pdb)
    if local.exists():
        return local
    pdb_id = wt_pdb.upper().replace(".PDB", "")
    out = results_dir / f"{pdb_id}.pdb"
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", out)
    return out


def plot_with_legend_bar(ax, labels, values, colors, title, ylabel):
    bars = ax.bar(labels, values, color=colors)
    for bar, lab in zip(bars, labels):
        bar.set_label(lab)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()


def add_structure_metrics(df: pd.DataFrame, device: torch.device, wt_coords: np.ndarray, max_n: int) -> pd.DataFrame:
    try:
        import esm
        from tmtools import tm_align
    except ModuleNotFoundError:
        df["plddt_score"] = np.nan
        df["tm_score"] = np.nan
        df["rmsd"] = np.nan
        return df

    model = esm.pretrained.esmfold_v1().eval().to(device)
    model.set_chunk_size(128)

    df = df.copy()
    df["plddt_score"] = np.nan
    df["tm_score"] = np.nan
    df["rmsd"] = np.nan

    order = df.sort_values("esm2_fitness", ascending=False).head(max_n).index
    with torch.no_grad():
        for idx in order:
            seq = str(df.loc[idx, "sequence"])
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

            df.loc[idx, "plddt_score"] = plddt
            df.loc[idx, "tm_score"] = tm_score
            df.loc[idx, "rmsd"] = rmsd

    return df


def evaluate_method(
    raw_csv: Path,
    method_name: str,
    model: torch.nn.Module,
    seq_len: int,
    esm2: ESM2Scorer,
    wt_seq: str,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    seq_col = infer_seq_col(raw)
    seqs = [clean_seq(str(s)) for s in raw[seq_col].tolist()]

    z = encode(model, seqs, seq_len, device, batch_size).to(device)
    with torch.no_grad():
        pldm = model.regressor_module(z).squeeze(-1).detach().cpu().numpy()

    esm_scores = []
    perplexities = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i : i + batch_size]
            s, p = esm2.score_and_perplexity(chunk)
            esm_scores.extend(s.detach().cpu().tolist())
            perplexities.extend(p.detach().cpu().tolist())

    return pd.DataFrame(
        {
            "sequence": seqs,
            "id": [seq_hash(s) for s in seqs],
            "method_name": method_name,
            "sequence_identity_wt": [identity(s, wt_seq) for s in seqs],
            "hamming_dist_wt": [hamming(s, wt_seq) for s in seqs],
            "esm2_fitness": esm_scores,
            "pldm_regressor_score": pldm,
            "perplexity": perplexities,
        }
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    proldm_root = (script_dir / args.proldm_root).resolve()
    train_csv = (proldm_root / args.train_csv).resolve()
    test_csv = (proldm_root / args.test_csv).resolve()
    ae_ckpt = (proldm_root / args.ae_ckpt).resolve()
    raw_dir = (script_dir / args.raw_dir).resolve()
    results_dir = (script_dir / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline_raw = raw_dir / "raw_baseline_pldm.csv"
    ess_raw = raw_dir / "raw_results_ess.csv"
    tess_raw = raw_dir / "raw_results_tess.csv"
    for p in [train_csv, test_csv, ae_ckpt, baseline_raw, ess_raw, tess_raw]:
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    wt_seq = choose_wt(train_df)
    seq_col = infer_seq_col(train_df)

    sys.path.insert(0, str(proldm_root))
    from model.JTAE.models_condif_1d import jtae

    ckpt_raw = torch.load(ae_ckpt, map_location=device)
    state = normalize_ckpt_state(ckpt_raw)
    dims = infer_dims_from_state(state)
    hparams = build_hparams(args.dataset, dims, device)
    model = jtae(hparams).to(device)
    model.load_state_dict(filter_by_shape(state, model.state_dict()), strict=False)
    model.eval()

    esm2 = ESM2Scorer(args.esm2_model, device, args.esm2_head_path)

    wt_score, wt_perp = esm2.score_and_perplexity([wt_seq])
    wt_reg = None
    with torch.no_grad():
        wt_z = encode(model, [wt_seq], dims["seq_len"], device, args.batch_size).to(device)
        wt_reg = model.regressor_module(wt_z).squeeze(-1).detach().cpu().item()
    wt_esm2 = float(wt_score.detach().cpu().item())
    wt_plddt = float("nan")
    if args.with_structure:
        try:
            import esm

            wt_pdb = ensure_pdb(args.wt_pdb, results_dir)
            wt_coords = parse_ca_coords_from_pdb(wt_pdb)
            fold_model = esm.pretrained.esmfold_v1().eval().to(device)
            fold_model.set_chunk_size(128)
            with torch.no_grad():
                wt_plddt = pdb_mean_plddt(fold_model.infer_pdb(wt_seq))
        except ModuleNotFoundError:
            wt_plddt = float("nan")

    baseline = evaluate_method(baseline_raw, "baseline_pldm", model, dims["seq_len"], esm2, wt_seq, device, args.batch_size)
    ess = evaluate_method(ess_raw, "ess", model, dims["seq_len"], esm2, wt_seq, device, args.batch_size)
    tess = evaluate_method(tess_raw, "tess", model, dims["seq_len"], esm2, wt_seq, device, args.batch_size)

    if args.with_structure:
        wt_pdb = ensure_pdb(args.wt_pdb, results_dir)
        wt_coords = parse_ca_coords_from_pdb(wt_pdb)
        baseline = add_structure_metrics(baseline, device, wt_coords, args.structure_max_per_method)
        ess = add_structure_metrics(ess, device, wt_coords, args.structure_max_per_method)
        tess = add_structure_metrics(tess, device, wt_coords, args.structure_max_per_method)
    else:
        for df in [baseline, ess, tess]:
            df["plddt_score"] = np.nan
            df["tm_score"] = np.nan
            df["rmsd"] = np.nan

    baseline.to_csv(results_dir / "baseline_pldm.csv", index=False)
    ess.to_csv(results_dir / "results_ess.csv", index=False)
    tess.to_csv(results_dir / "results_tess.csv", index=False)

    # Centerpiece performance table
    table = write_comparative_table(
        results_dir=results_dir,
        wt_seq=wt_seq,
        wt_identity_df=baseline,
        baseline=baseline,
        ess=ess,
        tess=tess,
        wt_plddt=wt_plddt,
        wt_esm2_score=wt_esm2,
        wt_reg_score=wt_reg,
    )

    # UMAP
    train_sample = train_df.sample(min(args.umap_train_samples, len(train_df)), random_state=args.seed)
    z_train = encode(model, [str(s) for s in train_sample[seq_col].tolist()], dims["seq_len"], device, args.batch_size)
    z_baseline = encode(model, baseline["sequence"].tolist(), dims["seq_len"], device, args.batch_size)
    z_ess = encode(model, ess["sequence"].tolist(), dims["seq_len"], device, args.batch_size)
    z_tess = encode(model, tess["sequence"].tolist(), dims["seq_len"], device, args.batch_size)

    all_z = torch.cat([z_train, z_baseline, z_ess, z_tess], dim=0).numpy()
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=args.seed).fit_transform(all_z)

    n0 = len(z_train)
    n1 = n0 + len(z_baseline)
    n2 = n1 + len(z_ess)
    sources = ["natural"] * len(z_train) + ["baseline_pldm"] * len(z_baseline) + ["ess"] * len(z_ess) + ["tess"] * len(z_tess)
    
    # Extract fluorescence values for natural proteins
    fit_col = infer_fitness_col(train_sample)
    fitness_vals = train_sample[fit_col].values if fit_col is not None else None
    
    umap_df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "source": sources})
    umap_df.to_csv(results_dir / "umap_points.csv", index=False)

    combo = pd.concat([baseline, ess, tess], ignore_index=True)

    # Shared specs for both UMAP variants
    legend_specs = [("baseline_pldm", "tab:orange", 0.8, 14), ("ess", "tab:blue", 0.8, 14), ("tess", "tab:red", 0.8, 14)]
    natural_d = umap_df[umap_df["source"] == "natural"]

    # UMAP with fluorescence heatmap on natural proteins
    plt.figure(figsize=(12, 8))
    for src, c, a, s in legend_specs:
        d = umap_df[umap_df["source"] == src]
        plt.scatter(d["x"], d["y"], c=c, alpha=a, s=s, label=src, edgecolors='none')
    if fitness_vals is not None:
        scatter = plt.scatter(natural_d["x"], natural_d["y"], c=fitness_vals, cmap="viridis", alpha=0.2, s=10,
                             label="natural", edgecolors='none')
        cbar = plt.colorbar(scatter)
        cbar.set_label("Fluorescence", rotation=270, labelpad=20)
    else:
        plt.scatter(natural_d["x"], natural_d["y"], c="lightgray", alpha=0.15, s=8, label="natural", edgecolors='none')
    plt.title("Latent Space UMAP (Natural = Fluorescence Heatmap)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_umap_latent_space.png", dpi=220)
    plt.savefig(results_dir / "plot_umap_latent_space_heatmap.png", dpi=220)
    plt.close()

    # UMAP without heatmap coloring on natural proteins
    plt.figure(figsize=(12, 8))
    for src, c, a, s in legend_specs:
        d = umap_df[umap_df["source"] == src]
        plt.scatter(d["x"], d["y"], c=c, alpha=a, s=s, label=src, edgecolors='none')
    plt.scatter(natural_d["x"], natural_d["y"], c="lightgray", alpha=0.15, s=8, label="natural", edgecolors='none')
    plt.title("Latent Space UMAP (Natural = Gray)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_umap_latent_space_no_heatmap.png", dpi=220)
    plt.close()

    # Additional quality diagnostics
    plot_stability_fitness(combo, results_dir)
    plot_perplexity_histogram(combo, results_dir)
    plot_aa_propensity_heatmap(combo, results_dir)
    holdout_seq_col = infer_seq_col(test_df)
    holdout_fit_col = infer_fitness_col(test_df)
    if holdout_fit_col is not None:
        plot_nearest_holdout_hamming_vs_fluorescence(
            generated_df=combo,
            holdout_df=test_df,
            holdout_seq_col=holdout_seq_col,
            holdout_fit_col=holdout_fit_col,
            results_dir=results_dir,
        )

    # Identity vs Fitness Pareto
    plt.figure(figsize=(9, 7))
    for m, c in [("baseline_pldm", "tab:blue"), ("ess", "tab:orange"), ("tess", "tab:green")]:
        d = combo[combo["method_name"] == m]
        plt.scatter(d["sequence_identity_wt"], d["esm2_fitness"], c=c, alpha=0.7, s=16, label=m)
    plt.xlabel("Sequence Identity to WT")
    plt.ylabel("ESM-2 Fitness")
    plt.title("Identity vs Fitness Pareto Front")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plot_identity_vs_fitness_pareto.png", dpi=220)
    plt.close()

    # KL divergence
    fit_col = infer_fitness_col(test_df)
    holdout = test_df.nlargest(100, fit_col) if fit_col is not None else test_df.head(100)
    hold_dist = aa_distribution([clean_seq(str(s)) for s in holdout[seq_col].tolist()])
    kl_vals = {
        "baseline_pldm": dkl(aa_distribution(baseline["sequence"].tolist()), hold_dist),
        "ess": dkl(aa_distribution(ess["sequence"].tolist()), hold_dist),
        "tess": dkl(aa_distribution(tess["sequence"].tolist()), hold_dist),
    }
    plt.figure(figsize=(8, 6))
    kl_labels = list(kl_vals.keys())
    kl_vals_list = list(kl_vals.values())
    bars = plt.bar(kl_labels, kl_vals_list, color=["tab:blue", "tab:orange", "tab:green"])
    for bar, lab in zip(bars, kl_labels):
        bar.set_label(lab)
    plt.legend()
    plt.ylabel("DKL to top-100 natural test holdout")
    plt.title("KL-Divergence Bar Chart")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_kl_divergence.png", dpi=220)
    plt.close()

    # Pearson correlation
    plt.figure(figsize=(8, 6))
    for m, c in [("baseline_pldm", "tab:blue"), ("ess", "tab:orange"), ("tess", "tab:green")]:
        d = combo[combo["method_name"] == m]
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
        "avg_plddt": {
            "baseline_pldm": float(np.nanmean(baseline["plddt_score"])),
            "ess": float(np.nanmean(ess["plddt_score"])),
            "tess": float(np.nanmean(tess["plddt_score"])),
        },
        "avg_tm": {
            "baseline_pldm": float(np.nanmean(baseline["tm_score"])),
            "ess": float(np.nanmean(ess["tm_score"])),
            "tess": float(np.nanmean(tess["tm_score"])),
        },
        "avg_rmsd": {
            "baseline_pldm": float(np.nanmean(baseline["rmsd"])),
            "ess": float(np.nanmean(ess["rmsd"])),
            "tess": float(np.nanmean(tess["rmsd"])),
        },
        "kl": kl_vals,
        "benchmark_table_path": str(results_dir / "table_fitness_fidelity_benchmark.csv"),
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved outputs to: {results_dir}")


if __name__ == "__main__":
    main()
