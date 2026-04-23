import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap

from tfg_analysis import build_jtae, encode_sequences, pdb_mean_plddt
from tfg_generation import ESM2Guidance, choose_start_sequence, infer_fitness_column, infer_seq_column, set_seed, str2bool


AA_VOCAB = list("ILVFMCAGPTSYWQNHEDKRX")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline PLDM vs ESS vs TESS.")
    parser.add_argument("--proldm-root", type=str, default="../PROLDM_OUTLIER")
    parser.add_argument("--checkpoint", type=str, default="train_logs/GFP/epoch_1000.pt")
    parser.add_argument("--train-csv", type=str, default="data/mut_data/GFP-train.csv")
    parser.add_argument("--baseline-raw-csv", type=str, default="generated_seq/raw_baseline_pldm.csv")
    parser.add_argument("--ess-raw-csv", type=str, default="generated_seq/raw_results_ess.csv")
    parser.add_argument("--tess-raw-csv", type=str, default="generated_seq/raw_results_tess.csv")
    parser.add_argument("--dataset", type=str, default="GFP")
    parser.add_argument("--umap-train-samples", type=int, default=5000)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--esm2-model", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--esm2-head-path", type=str, default=None)
    parser.add_argument("--with-structure", type=str2bool, default=False)
    parser.add_argument("--structure-top-k-per-method", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--baseline-diff-steps", type=int, default=500)
    parser.add_argument("--results-dir", type=str, default="results/comparison_study")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sequence_id(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:12]


def identity(seq1: str, seq2: str) -> float:
    n = min(len(seq1), len(seq2))
    if n == 0:
        return 0.0
    return float(sum(1 for a, b in zip(seq1[:n], seq2[:n]) if a == b) / n)


def hamming_dist(seq1: str, seq2: str) -> int:
    n = min(len(seq1), len(seq2))
    dist = sum(1 for a, b in zip(seq1[:n], seq2[:n]) if a != b)
    dist += abs(len(seq1) - len(seq2))
    return int(dist)


def normalize_sequence_col(frame: pd.DataFrame) -> pd.Series:
    for col in ["sequence_trimmed", "sequence", "pred_seq", "seq", "primary", "protein_sequence"]:
        if col in frame.columns:
            return frame[col].astype(str).str.replace("J", "", regex=False).str.replace("*", "", regex=False).str.replace("-", "", regex=False)
    raise ValueError("Unable to find sequence column in input CSV.")


def score_sequences(
    seqs: List[str],
    model: torch.nn.Module,
    seq_len: int,
    esm2: ESM2Guidance,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    z = encode_sequences(model, seqs, seq_len, device, batch_size).to(device)
    with torch.no_grad():
        pldm_scores = model.regressor_module(z).squeeze(-1).detach().cpu().numpy()

    esm_scores: List[float] = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i : i + batch_size]
            esm_scores.extend(esm2.score(chunk).detach().cpu().tolist())

    return pldm_scores, np.asarray(esm_scores), z.detach().cpu()


def amino_freq_dist(seqs: List[str]) -> np.ndarray:
    counts = np.zeros(len(AA_VOCAB), dtype=np.float64)
    idx = {aa: i for i, aa in enumerate(AA_VOCAB)}
    total = 0
    for seq in seqs:
        for ch in seq:
            mapped = ch if ch in idx else "X"
            counts[idx[mapped]] += 1
            total += 1
    probs = (counts + 1e-8) / (total + 1e-8 * len(AA_VOCAB))
    return probs


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))


def compute_plddt_for_top(
    frame: pd.DataFrame,
    top_k: int,
    device: torch.device,
) -> pd.Series:
    try:
        import esm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing ESMFold dependency for pLDDT scoring. Run 'uv sync' or set --with-structure false."
        ) from exc

    model = esm.pretrained.esmfold_v1().eval().to(device)
    model.set_chunk_size(128)

    out = pd.Series(np.nan, index=frame.index, dtype=float)
    top_idx = frame.sort_values("esm2_fitness", ascending=False).head(top_k).index
    with torch.no_grad():
        for idx in top_idx:
            seq = str(frame.loc[idx, "sequence"])
            pdb_text = model.infer_pdb(seq)
            out.loc[idx] = pdb_mean_plddt(pdb_text)
    return out


def prepare_method_df(
    raw: pd.DataFrame,
    method_name: str,
    wt_seq: str,
    model: torch.nn.Module,
    seq_len: int,
    esm2: ESM2Guidance,
    device: torch.device,
    batch_size: int,
    with_structure: bool,
    structure_top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seqs = normalize_sequence_col(raw).tolist()
    pldm_scores, esm_scores, latents = score_sequences(seqs, model, seq_len, esm2, device, batch_size)

    out = pd.DataFrame(
        {
            "sequence": seqs,
            "id": [sequence_id(s) for s in seqs],
            "method_name": method_name,
            "hamming_dist_wt": [hamming_dist(s, wt_seq) for s in seqs],
            "plddt_score": np.nan,
            "esm2_fitness": esm_scores,
            "pldm_regressor_score": pldm_scores,
        }
    )

    if with_structure:
        out["plddt_score"] = compute_plddt_for_top(out.rename(columns={"sequence": "sequence"}), structure_top_k, device)

    trace = pd.DataFrame({
        "method_name": method_name,
        "sequence": seqs,
        "esm2_fitness": esm_scores,
        "pldm_regressor_score": pldm_scores,
        "latent_0": latents[:, 0].numpy(),
    })

    if "attempts" in raw.columns:
        trace["eval_steps"] = raw["attempts"].fillna(1).astype(float).to_numpy()
    elif "step" in raw.columns:
        trace["eval_steps"] = np.ones(len(raw), dtype=float)
    else:
        trace["eval_steps"] = np.ones(len(raw), dtype=float)

    return out, trace


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    proldm_root = (script_dir / args.proldm_root).resolve()
    checkpoint_path = (proldm_root / args.checkpoint).resolve()
    train_csv = (proldm_root / args.train_csv).resolve()
    baseline_raw_csv = (script_dir / args.baseline_raw_csv).resolve()
    ess_raw_csv = (script_dir / args.ess_raw_csv).resolve()
    tess_raw_csv = (script_dir / args.tess_raw_csv).resolve()
    results_dir = (script_dir / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    for path in [checkpoint_path, train_csv, baseline_raw_csv, ess_raw_csv, tess_raw_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    train_df = pd.read_csv(train_csv)
    train_seq_col = infer_seq_column(train_df)
    wt_seq = choose_start_sequence(train_df, "wildtype", 0)

    model, seq_len = build_jtae(proldm_root, checkpoint_path, args.dataset, device)
    esm2 = ESM2Guidance(args.esm2_model, device, args.esm2_head_path)

    baseline_raw = pd.read_csv(baseline_raw_csv)
    ess_raw = pd.read_csv(ess_raw_csv)
    tess_raw = pd.read_csv(tess_raw_csv)

    baseline_df, baseline_trace = prepare_method_df(
        baseline_raw,
        "baseline_pldm",
        wt_seq,
        model,
        seq_len,
        esm2,
        device,
        args.score_batch_size,
        args.with_structure,
        args.structure_top_k_per_method,
    )
    baseline_trace["eval_steps"] = float(args.baseline_diff_steps)

    ess_df, ess_trace = prepare_method_df(
        ess_raw,
        "ess",
        wt_seq,
        model,
        seq_len,
        esm2,
        device,
        args.score_batch_size,
        args.with_structure,
        args.structure_top_k_per_method,
    )
    tess_df, tess_trace = prepare_method_df(
        tess_raw,
        "tess",
        wt_seq,
        model,
        seq_len,
        esm2,
        device,
        args.score_batch_size,
        args.with_structure,
        args.structure_top_k_per_method,
    )

    baseline_df.to_csv(results_dir / "baseline_pldm.csv", index=False)
    ess_df.to_csv(results_dir / "results_ess.csv", index=False)
    tess_df.to_csv(results_dir / "results_tess.csv", index=False)

    if len(train_df) > args.umap_train_samples:
        train_sample = train_df.sample(args.umap_train_samples, random_state=args.seed).copy()
    else:
        train_sample = train_df.copy()

    train_seqs = train_sample[train_seq_col].astype(str).tolist()
    z_train = encode_sequences(model, train_seqs, seq_len, device, args.score_batch_size)
    z_baseline = encode_sequences(model, baseline_df["sequence"].tolist(), seq_len, device, args.score_batch_size)
    z_ess = encode_sequences(model, ess_df["sequence"].tolist(), seq_len, device, args.score_batch_size)
    z_tess = encode_sequences(model, tess_df["sequence"].tolist(), seq_len, device, args.score_batch_size)

    all_z = torch.cat([z_train, z_baseline, z_ess, z_tess], dim=0).numpy()
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=args.seed)
    xy = reducer.fit_transform(all_z)

    n_train = len(z_train)
    n_baseline = len(z_baseline)
    n_ess = len(z_ess)
    n_tess = len(z_tess)
    split1 = n_train
    split2 = split1 + n_baseline
    split3 = split2 + n_ess

    plot_df = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "source": (["natural"] * n_train) + (["baseline_pldm"] * n_baseline) + (["ess"] * n_ess) + (["tess"] * n_tess),
    })
    plot_df.to_csv(results_dir / "umap_overlay_points.csv", index=False)

    plt.figure(figsize=(10, 8))
    nat = plot_df[plot_df["source"] == "natural"]
    plt.scatter(nat["x"], nat["y"], c="lightgray", s=8, alpha=0.35, label="GFP train")
    for source, color in [("baseline_pldm", "tab:blue"), ("ess", "tab:orange"), ("tess", "tab:green")]:
        d = plot_df[plot_df["source"] == source]
        plt.scatter(d["x"], d["y"], s=14, alpha=0.8, label=source, c=color)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("Latent Space Traversal: Natural vs Baseline vs ESS vs TESS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plot_umap_comparison.png", dpi=220)
    plt.close()

    fit_col = infer_fitness_column(train_df)
    centroid_metrics = {}
    if fit_col is not None:
        top_n = max(1, int(0.01 * len(train_df)))
        top_nat = train_df.nlargest(top_n, fit_col)
        z_top = encode_sequences(model, top_nat[train_seq_col].astype(str).tolist(), seq_len, device, args.score_batch_size).numpy()
        top_centroid = z_top.mean(axis=0)

        centroid_metrics = {
            "baseline_pldm": float(np.linalg.norm(z_baseline.numpy().mean(axis=0) - top_centroid)),
            "ess": float(np.linalg.norm(z_ess.numpy().mean(axis=0) - top_centroid)),
            "tess": float(np.linalg.norm(z_tess.numpy().mean(axis=0) - top_centroid)),
        }

    novelty = pd.concat([baseline_df, ess_df, tess_df], ignore_index=True)
    novelty.to_csv(results_dir / "novelty_stability_points.csv", index=False)

    plt.figure(figsize=(9, 7))
    for method, color in [("baseline_pldm", "tab:blue"), ("ess", "tab:orange"), ("tess", "tab:green")]:
        d = novelty[novelty["method_name"] == method]
        plt.scatter(d["hamming_dist_wt"], d["plddt_score"], label=method, alpha=0.75, s=16, c=color)
    plt.axhline(70.0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Hamming Distance from WT")
    plt.ylabel("pLDDT Score")
    plt.title("Novelty-Stability Tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plot_novelty_vs_stability.png", dpi=220)
    plt.close()

    if fit_col is None:
        holdout = train_df.head(100)
    else:
        holdout = train_df.nlargest(100, fit_col)
    holdout_dist = amino_freq_dist(holdout[train_seq_col].astype(str).tolist())

    kls = {
        "baseline_pldm": kl_div(amino_freq_dist(baseline_df["sequence"].tolist()), holdout_dist),
        "ess": kl_div(amino_freq_dist(ess_df["sequence"].tolist()), holdout_dist),
        "tess": kl_div(amino_freq_dist(tess_df["sequence"].tolist()), holdout_dist),
    }
    kls["delta_kl_ess_vs_baseline"] = kls["ess"] - kls["baseline_pldm"]
    kls["delta_kl_tess_vs_baseline"] = kls["tess"] - kls["baseline_pldm"]

    plt.figure(figsize=(8, 6))
    keys = ["baseline_pldm", "ess", "tess"]
    vals = [kls[k] for k in keys]
    plt.bar(keys, vals, color=["tab:blue", "tab:orange", "tab:green"])
    plt.ylabel("KL divergence to top-100 natural holdout")
    plt.title("Distributional Fidelity (Lower is better)")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_kl_divergence.png", dpi=220)
    plt.close()

    traces = pd.concat([baseline_trace, ess_trace, tess_trace], ignore_index=True)
    traces.to_csv(results_dir / "convergence_trace_points.csv", index=False)

    plt.figure(figsize=(9, 7))
    for method, color in [("baseline_pldm", "tab:blue"), ("ess", "tab:orange"), ("tess", "tab:green")]:
        d = traces[traces["method_name"] == method].copy()
        d = d.reset_index(drop=True)
        d["eval_budget"] = d["eval_steps"].cumsum()
        d["cum_best_esm2"] = d["esm2_fitness"].cummax()
        plt.plot(d["eval_budget"], d["cum_best_esm2"], label=method, linewidth=2, color=color)
    plt.xlabel("Number of steps / evaluations")
    plt.ylabel("Best-so-far ESM2 fitness")
    plt.title("Sampling Convergence Speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "plot_convergence_speed.png", dpi=220)
    plt.close()

    summary = {
        "num_sequences": {
            "baseline_pldm": int(len(baseline_df)),
            "ess": int(len(ess_df)),
            "tess": int(len(tess_df)),
        },
        "centroid_distance_to_top1pct": centroid_metrics,
        "kl_divergence": kls,
        "with_structure": bool(args.with_structure),
    }
    with open(results_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved comparison study artifacts to: {results_dir}")


if __name__ == "__main__":
    main()
