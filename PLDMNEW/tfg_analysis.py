import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from Bio.PDB import PDBParser
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from tfg_generation import (
    ESM2Guidance,
    choose_start_sequence,
    extract_ae_state_dict,
    filter_state_by_shape,
    get_jtae_hparams,
    infer_model_dims_from_state_dict,
    infer_seq_column,
    seq_to_inds,
    set_seed,
    str2bool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TFG post-generation analysis pipeline.")
    parser.add_argument("--proldm-root", type=str, default="../PROLDM_OUTLIER")
    parser.add_argument("--checkpoint", type=str, default="train_logs/GFP/epoch_1000.pt")
    parser.add_argument("--train-csv", type=str, default="data/mut_data/GFP-train.csv")
    parser.add_argument("--generated-csv", type=str, default="generated_seq/tfg_generation_1000.csv")
    parser.add_argument("--dataset", type=str, default="GFP")
    parser.add_argument("--umap-train-samples", type=int, default=5000)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--esm2-model", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--esm2-head-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--with-structure", type=str2bool, default=False)
    parser.add_argument("--structure-top-k", type=int, default=100)
    parser.add_argument("--wt-pdb", type=str, default="1GFL")
    parser.add_argument("--results-dir", type=str, default="results/full_analysis")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalize_checkpoint_state(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        ckpt_raw = ckpt_obj["state_dict"]
    elif isinstance(ckpt_obj, dict):
        ckpt_raw = ckpt_obj
    else:
        raise ValueError("Unsupported checkpoint format")

    normalized = {}
    for key, val in ckpt_raw.items():
        new_key = key
        for prefix in ("module.jtae.", "jtae.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = val
    return normalized


def build_jtae(
    proldm_root: Path,
    checkpoint_path: Path,
    dataset: str,
    requested_device: torch.device,
) -> Tuple[torch.nn.Module, int]:
    sys.path.insert(0, str(proldm_root))
    from model.JTAE.models_condif_1d import jtae

    ckpt_obj = torch.load(checkpoint_path, map_location=requested_device)
    normalized_raw = _normalize_checkpoint_state(ckpt_obj)
    dims = infer_model_dims_from_state_dict(normalized_raw)

    hparams = get_jtae_hparams(
        seq_len=dims["seq_len"],
        dataset=dataset,
        batch_size=16,
        latent_dim=dims["latent_dim"],
        input_dim=dims["input_dim"],
        hidden_dim=dims["hidden_dim"],
        embedding_dim=dims["embedding_dim"],
        device=requested_device,
    )
    model = jtae(hparams).to(requested_device)
    model.eval()

    ae_state = extract_ae_state_dict(ckpt_obj, model.state_dict().keys())
    ae_state = filter_state_by_shape(ae_state, model.state_dict())
    model.load_state_dict(ae_state, strict=False)
    return model, dims["seq_len"]


def encode_sequences(
    model: torch.nn.Module,
    sequences: List[str],
    seq_len: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    latents: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            inds = torch.stack([seq_to_inds(seq, seq_len) for seq in batch], dim=0).to(device)
            z = model.encode(inds).to(torch.float32)
            latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def calculate_identity(seq1: str, seq2: str) -> float:
    n = min(len(seq1), len(seq2))
    if n == 0:
        return 0.0
    return float(sum(1 for a, b in zip(seq1[:n], seq2[:n]) if a == b) / n)


def ensure_pdb_file(wt_pdb: str, results_dir: Path) -> Path:
    candidate = Path(wt_pdb)
    if candidate.exists():
        return candidate

    pdb_id = wt_pdb.upper().replace(".PDB", "")
    out_path = results_dir / f"{pdb_id}.pdb"
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    urllib.request.urlretrieve(url, out_path)
    return out_path


def parse_ca_coords(pdb_path: Path) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    coords = []
    for atom in structure.get_atoms():
        if atom.get_name() == "CA":
            coords.append(atom.get_coord())
    if len(coords) == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    return np.asarray(coords, dtype=np.float64)


def pdb_mean_plddt(pdb_text: str) -> float:
    vals: List[float] = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            try:
                vals.append(float(line[60:66].strip()))
            except ValueError:
                continue
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def structure_validation(
    frame: pd.DataFrame,
    wt_pdb_path: Path,
    top_k: int,
    device: torch.device,
    out_dir: Path,
) -> pd.DataFrame:
    try:
        import esm
        from tmtools import tm_align
    except ModuleNotFoundError as exc:
        missing_mod = getattr(exc, "name", "<unknown>")
        raise ModuleNotFoundError(
            f"Missing dependency '{missing_mod}' required for structural validation. "
            "Run 'uv sync' in PLDMNEW to install dependencies (including omegaconf/esm/tmtools), "
            "or run analysis with --with-structure false."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    wt_coords = parse_ca_coords(wt_pdb_path)

    model = esm.pretrained.esmfold_v1().eval().to(device)
    model.set_chunk_size(128)

    top = frame.sort_values("proldm_score", ascending=False).head(top_k).copy()
    rows = []

    with torch.no_grad():
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            seq = str(row["sequence_trimmed"])
            pdb_text = model.infer_pdb(seq)
            pdb_path = out_dir / f"design_{rank:03d}.pdb"
            pdb_path.write_text(pdb_text)

            pred_coords = parse_ca_coords(pdb_path)
            n = min(len(wt_coords), len(pred_coords))
            align = tm_align(pred_coords[:n], wt_coords[:n], seq[:n], seq[:n])

            rows.append(
                {
                    "rank": rank,
                    "sequence_trimmed": seq,
                    "identity_to_wt": float(row["identity_to_wt"]),
                    "proldm_score": float(row["proldm_score"]),
                    "esm2_score": float(row["esm2_score"]),
                    "plddt_mean": pdb_mean_plddt(pdb_text),
                    "tm_score": float(align.tm_norm_chain1),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "structural_metrics_top100.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(result["identity_to_wt"], result["plddt_mean"], c=result["tm_score"], cmap="viridis", s=40)
    plt.xlabel("Sequence Identity to WT")
    plt.ylabel("pLDDT")
    plt.title("Structural Fidelity: pLDDT vs Sequence Identity")
    cbar = plt.colorbar()
    cbar.set_label("TM-score")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_structural_plddt_vs_identity.png", dpi=220)
    plt.close()

    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    proldm_root = (script_dir / args.proldm_root).resolve()
    checkpoint_path = (proldm_root / args.checkpoint).resolve()
    train_csv = (proldm_root / args.train_csv).resolve()
    generated_csv = (script_dir / args.generated_csv).resolve()
    results_dir = (script_dir / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not train_csv.exists():
        raise FileNotFoundError(f"Train csv not found: {train_csv}")
    if not generated_csv.exists():
        raise FileNotFoundError(f"Generated csv not found: {generated_csv}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    train_df = pd.read_csv(train_csv)
    gen_df = pd.read_csv(generated_csv)
    if "sequence_trimmed" not in gen_df.columns and "sequence" in gen_df.columns:
        gen_df["sequence_trimmed"] = gen_df["sequence"].astype(str).str.replace("J", "", regex=False)

    wt_seq = choose_start_sequence(train_df, "wildtype", 0)
    train_seq_col = infer_seq_column(train_df)

    model, seq_len = build_jtae(
        proldm_root=proldm_root,
        checkpoint_path=checkpoint_path,
        dataset=args.dataset,
        requested_device=device,
    )

    if len(train_df) > args.umap_train_samples:
        train_sample = train_df.sample(args.umap_train_samples, random_state=args.seed).copy()
    else:
        train_sample = train_df.copy()

    train_sequences = train_sample[train_seq_col].astype(str).tolist()
    gen_sequences = gen_df["sequence_trimmed"].astype(str).tolist()

    z_train = encode_sequences(model, train_sequences, seq_len, device, args.score_batch_size)
    z_gen = encode_sequences(model, gen_sequences, seq_len, device, args.score_batch_size)

    umap_input = torch.cat([z_train, z_gen], dim=0).numpy()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.seed,
    )
    umap_xy = reducer.fit_transform(umap_input)

    train_labels = train_sample["label"].astype(int).tolist() if "label" in train_sample.columns else [-1] * len(train_sample)
    plot_df = pd.DataFrame(
        {
            "x": umap_xy[:, 0],
            "y": umap_xy[:, 1],
            "source": ["natural_train"] * len(train_sample) + ["ess_generated"] * len(gen_df),
            "label": train_labels + [9] * len(gen_df),
        }
    )
    plot_df.to_csv(results_dir / "umap_points.csv", index=False)

    plt.figure(figsize=(9, 7))
    nat = plot_df[plot_df["source"] == "natural_train"]
    gen = plot_df[plot_df["source"] == "ess_generated"]
    plt.scatter(nat["x"], nat["y"], c=nat["label"], cmap="tab10", alpha=0.55, s=12)
    plt.scatter(gen["x"], gen["y"], c="black", alpha=0.9, s=14, label="ESS/TESS generated")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("Latent Space UMAP: Natural GFP + ESS/TESS Samples")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_umap_latent_map.png", dpi=220)
    plt.close()

    gen_df["identity_to_wt"] = [calculate_identity(str(seq), wt_seq) for seq in gen_df["sequence_trimmed"].astype(str)]
    gen_df["hamming_to_wt"] = [int(round((1.0 - x) * len(wt_seq))) for x in gen_df["identity_to_wt"]]

    plt.figure(figsize=(9, 6))
    plt.hist(gen_df["identity_to_wt"], bins=30, alpha=0.9)
    plt.xlabel("Sequence Identity to WT")
    plt.ylabel("Count")
    plt.title("Novelty: Identity Distribution vs WT GFP")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_identity_histogram.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.hist(gen_df["hamming_to_wt"], bins=30, alpha=0.9)
    plt.xlabel("Hamming Distance to WT")
    plt.ylabel("Count")
    plt.title("Novelty: Hamming Distance Distribution vs WT GFP")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_hamming_histogram.png", dpi=220)
    plt.close()

    z_gen_for_scores = encode_sequences(model, gen_sequences, seq_len, device, args.score_batch_size).to(device)
    with torch.no_grad():
        proldm_scores = model.regressor_module(z_gen_for_scores).squeeze(-1).detach().cpu().numpy()

    esm2 = ESM2Guidance(args.esm2_model, device=device, head_path=args.esm2_head_path)
    esm_scores: List[float] = []
    with torch.no_grad():
        for i in range(0, len(gen_sequences), args.score_batch_size):
            batch = gen_sequences[i : i + args.score_batch_size]
            scores = esm2.score(batch).detach().cpu().tolist()
            esm_scores.extend(scores)

    gen_df["proldm_score"] = proldm_scores
    gen_df["esm2_score"] = np.asarray(esm_scores)

    r2 = float(r2_score(gen_df["proldm_score"], gen_df["esm2_score"]))
    pear = float(pearsonr(gen_df["proldm_score"], gen_df["esm2_score"])[0])

    plt.figure(figsize=(8, 6))
    plt.scatter(gen_df["esm2_score"], gen_df["proldm_score"], s=16, alpha=0.7)
    plt.xlabel("ESM-2 score")
    plt.ylabel("PRO-LDM regressor score")
    plt.title(f"Cross-Model Fitness Validation (Pearson={pear:.3f}, R2={r2:.3f})")
    plt.tight_layout()
    plt.savefig(results_dir / "plot_cross_model_correlation.png", dpi=220)
    plt.close()

    structural_summary = {"status": "skipped"}
    if args.with_structure:
        wt_pdb_path = ensure_pdb_file(args.wt_pdb, results_dir)
        struct_df = structure_validation(
            frame=gen_df,
            wt_pdb_path=wt_pdb_path,
            top_k=args.structure_top_k,
            device=device,
            out_dir=results_dir / "structures",
        )
        structural_summary = {
            "status": "completed",
            "top_k": int(len(struct_df)),
            "pct_plddt_gt_70": float((struct_df["plddt_mean"] > 70.0).mean()),
            "pct_tm_gt_0_5": float((struct_df["tm_score"] > 0.5).mean()),
        }

    gen_df.to_csv(results_dir / "generated_with_scores.csv", index=False)

    summary = {
        "n_generated": int(len(gen_df)),
        "n_train_used_for_umap": int(len(train_sample)),
        "pearson_cross_model": pear,
        "r2_cross_model": r2,
        "identity_mean": float(gen_df["identity_to_wt"].mean()),
        "identity_min": float(gen_df["identity_to_wt"].min()),
        "identity_max": float(gen_df["identity_to_wt"].max()),
        "hamming_mean": float(gen_df["hamming_to_wt"].mean()),
        "structural": structural_summary,
    }
    with open(results_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved analysis outputs to: {results_dir}")


if __name__ == "__main__":
    main()
