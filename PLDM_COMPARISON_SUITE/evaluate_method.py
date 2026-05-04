import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

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


# ---------------------------------------------------------------------------
# Amino-acid alphabet
# ---------------------------------------------------------------------------

_AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical, alphabetical
_AA_TO_IDX = {a: i for i, a in enumerate(_AA_ALPHABET)}
_GAP_IDX = len(_AA_ALPHABET)  # index for unknown / gap


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
    p.add_argument("--runtime-sec", type=float, default=None)
    # New: LOO calibration sample size (set 0 to skip)
    p.add_argument("--loo-max-samples", type=int, default=200,
                   help="Max training sequences used for LOO regressor calibration (0 = skip).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------

def encode(model, seqs: List[str], seq_len: int, device: torch.device, batch_size: int) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i: i + batch_size]
            inds = torch.stack([seq_to_inds(s, seq_len) for s in chunk], dim=0).to(device)
            z = model.encode(inds).to(torch.float32)
            out.append(z.cpu())
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Novelty metrics
# ---------------------------------------------------------------------------

def min_train_distance_metrics(
    seq: str,
    train_seqs: List[str],
) -> Tuple[float, int, str]:
    """Return (best_identity, min_hamming, nearest_train_seq)."""
    best_id = -1.0
    best_h = int(1e9)
    best_match = ""
    for t in train_seqs:
        id_ = identity(seq, t)
        h = hamming(seq, t)
        if id_ > best_id:
            best_id = id_
            best_h = h
            best_match = t
    return float(best_id), int(best_h), best_match


# ---------------------------------------------------------------------------
# Per-position amino-acid distribution helpers
# ---------------------------------------------------------------------------

def _encode_positions(seqs: List[str]) -> np.ndarray:
    """Return (N, L) integer array; unknown residues mapped to _GAP_IDX."""
    L = max(len(s) for s in seqs) if seqs else 1
    arr = np.full((len(seqs), L), _GAP_IDX, dtype=np.int32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s[:L]):
            arr[i, j] = _AA_TO_IDX.get(ch, _GAP_IDX)
    return arr


def positional_entropy(seqs: List[str]) -> np.ndarray:
    """Shannon entropy at each position (nats), ignoring gap/unknown."""
    arr = _encode_positions(seqs)
    N, L = arr.shape
    H = np.zeros(L, dtype=np.float64)
    for j in range(L):
        col = arr[:, j]
        valid = col[col < _GAP_IDX]
        if len(valid) == 0:
            H[j] = 0.0
            continue
        counts = np.bincount(valid, minlength=len(_AA_ALPHABET)).astype(np.float64)
        p = counts / counts.sum()
        nz = p > 0
        H[j] = -(p[nz] * np.log(p[nz])).sum()
    return H


def entropy_mse(train_seqs: List[str], gen_seqs: List[str]) -> float:
    """
    Mean-squared difference in per-position Shannon entropy between the
    training set and the generated set.  Lower = generated set has a
    similar positional diversity profile to training.
    """
    if not train_seqs or not gen_seqs:
        return float("nan")
    Ht = positional_entropy(train_seqs)
    Hg = positional_entropy(gen_seqs)
    L = max(len(Ht), len(Hg))
    Ht = np.pad(Ht, (0, L - len(Ht)))
    Hg = np.pad(Hg, (0, L - len(Hg)))
    return float(np.mean((Ht - Hg) ** 2))


def pairwise_residue_correlation(train_seqs: List[str], gen_seqs: List[str]) -> float:
    """
    Pearson correlation between the upper-triangle of the pairwise
    same-residue co-occurrence matrices of training and generated sets.
    +1.0 = identical pairwise residue structure.
    """
    if not train_seqs or not gen_seqs:
        return float("nan")

    At = _encode_positions(train_seqs)
    Ag = _encode_positions(gen_seqs)
    L = max(At.shape[1], Ag.shape[1])

    def _pad(X):
        if X.shape[1] < L:
            X = np.hstack([X, np.full((X.shape[0], L - X.shape[1]), _GAP_IDX, dtype=np.int32)])
        return X

    At = _pad(At)
    Ag = _pad(Ag)

    def _cooc_upper(X):
        vals = []
        for i in range(L):
            for j in range(i + 1, min(L, i + 51)):
                f_t = np.mean(X[:, i] == X[:, j])
                vals.append(f_t)
        return np.array(vals, dtype=np.float64)

    vt = _cooc_upper(At)
    vg = _cooc_upper(Ag)
    if vt.std() < 1e-12 or vg.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(vt, vg)[0, 1])


# ---------------------------------------------------------------------------
# NEW: Per-position KL divergence  D_KL(p_gen || p_train)
# ---------------------------------------------------------------------------

def positional_aa_freqs(seqs: List[str], L: int, pseudocount: float = 0.5) -> np.ndarray:
    """
    Return (L, 20) matrix of amino-acid frequencies at each position.
    Laplace (add-pseudocount) smoothing avoids zero-probability positions.
    Only the 20 canonical AAs are counted; gaps/unknowns are ignored.
    """
    counts = np.zeros((L, len(_AA_ALPHABET)), dtype=np.float64) + pseudocount
    for s in seqs:
        for j, ch in enumerate(s[:L]):
            idx = _AA_TO_IDX.get(ch, None)
            if idx is not None:
                counts[j, idx] += 1.0
    freqs = counts / counts.sum(axis=1, keepdims=True)
    return freqs


def positional_kl_divergence(
    train_seqs: List[str],
    gen_seqs: List[str],
) -> Tuple[float, np.ndarray]:
    """
    Per-position KL divergence D_KL(p_gen || p_train) averaged over positions.

    Returns
    -------
    mean_kl   : float  -- scalar summary (lower is better; 0 = identical)
    per_pos   : (L,) ndarray -- per-position KL values for inspection

    Notes
    -----
    Laplace smoothing (pseudocount=0.5) ensures p_train > 0 everywhere so
    D_KL is always finite.  We use D_KL(gen || train) so that the divergence
    penalises generated positions that place mass on residues the training
    set never uses.
    """
    if not train_seqs or not gen_seqs:
        return float("nan"), np.array([])

    L = max(max(len(s) for s in train_seqs), max(len(s) for s in gen_seqs))
    p_train = positional_aa_freqs(train_seqs, L)   # (L, 20)
    p_gen   = positional_aa_freqs(gen_seqs,   L)   # (L, 20)

    # D_KL(p_gen || p_train) = sum_a p_gen(a) * log(p_gen(a) / p_train(a))
    ratio     = np.log(p_gen / p_train)             # (L, 20)
    kl_per_pos = (p_gen * ratio).sum(axis=1)        # (L,)
    mean_kl   = float(np.mean(kl_per_pos))
    return mean_kl, kl_per_pos


# ---------------------------------------------------------------------------
# NEW: LOO regressor calibration
# ---------------------------------------------------------------------------

def loo_regressor_calibration(
    model,
    train_df: pd.DataFrame,
    seq_col: str,
    fitness_col: Optional[str],
    seq_len: int,
    device: torch.device,
    max_samples: int = 200,
) -> Dict[str, float]:
    """
    Leave-One-Out calibration of the PLDM regressor on the training set.

    For each held-out sequence i, the regressor predicts its fitness using
    only the *loaded* (already-trained) model weights -- we do NOT retrain.
    This is therefore a *retrospective* LOO check: it measures whether the
    regressor's predictions on individual training sequences correlate with
    ground-truth fitness labels, which calibrates how much we should trust
    its scores on novel generated sequences.

    Returns a dict with keys:
        loo_spearman_rho   : Spearman rank correlation (pred vs truth)
        loo_mse            : Mean squared error
        loo_mae            : Mean absolute error
        loo_n              : number of sequences evaluated

    If no ground-truth fitness column is found, all values are NaN.
    """
    empty = {"loo_spearman_rho": float("nan"),
             "loo_mse":          float("nan"),
             "loo_mae":          float("nan"),
             "loo_n":            0}

    if fitness_col is None or fitness_col not in train_df.columns:
        return empty
    if max_samples <= 0:
        return empty

    sub = train_df[[seq_col, fitness_col]].dropna().reset_index(drop=True)
    if len(sub) == 0:
        return empty

    # Subsample if large
    if len(sub) > max_samples:
        sub = sub.sample(n=max_samples, random_state=0).reset_index(drop=True)

    seqs   = [clean_seq(str(s)) for s in sub[seq_col].tolist()]
    truths = sub[fitness_col].to_numpy(dtype=np.float64)

    # Encode all at once and score with the frozen regressor
    z_all = encode(model, seqs, seq_len, device, batch_size=64).to(device)
    with torch.no_grad():
        preds = model.regressor_module(z_all).squeeze(-1).detach().cpu().numpy()

    valid = np.isfinite(preds) & np.isfinite(truths)
    if valid.sum() < 4:
        return empty

    rho, _ = spearmanr(preds[valid], truths[valid])
    mse    = float(np.mean((preds[valid] - truths[valid]) ** 2))
    mae    = float(np.mean(np.abs(preds[valid] - truths[valid])))

    return {
        "loo_spearman_rho": float(rho),
        "loo_mse":          mse,
        "loo_mae":          mae,
        "loo_n":            int(valid.sum()),
    }


# ---------------------------------------------------------------------------
# NEW: Spearman rho between regressor scores and true DMS brightness
# ---------------------------------------------------------------------------

def spearman_regressor_vs_truth(
    gen_seqs:      List[str],
    gen_scores:    np.ndarray,
    train_df:      pd.DataFrame,
    seq_col:       str,
    fitness_col:   Optional[str],
    identity_threshold: float = 0.95,
) -> Dict[str, float]:
    """
    Compute Spearman rho between the PLDM regressor scores for generated
    sequences and the ground-truth DMS fitness values for those same
    sequences (matched by sequence identity >= identity_threshold).

    This answers: "When a generated sequence closely matches a known training
    sequence, does the regressor rank them correctly?"

    Returns
    -------
    dict with keys:
        spearman_rho_vs_truth  : float
        n_matched              : int  -- how many gen seqs matched a training seq
    """
    empty = {"spearman_rho_vs_truth": float("nan"), "n_matched": 0}

    if fitness_col is None or fitness_col not in train_df.columns:
        return empty

    train_seqs_all    = [clean_seq(str(s)) for s in train_df[seq_col].dropna().tolist()]
    train_fitness_all = train_df[fitness_col].dropna().to_numpy(dtype=np.float64)

    if len(train_seqs_all) != len(train_fitness_all):
        # align by index after dropna
        tmp = train_df[[seq_col, fitness_col]].dropna().reset_index(drop=True)
        train_seqs_all    = [clean_seq(str(s)) for s in tmp[seq_col].tolist()]
        train_fitness_all = tmp[fitness_col].to_numpy(dtype=np.float64)

    matched_pred, matched_truth = [], []
    for seq, score in zip(gen_seqs, gen_scores):
        for t_seq, t_fit in zip(train_seqs_all, train_fitness_all):
            if identity(seq, t_seq) >= identity_threshold:
                matched_pred.append(score)
                matched_truth.append(t_fit)
                break  # take the first match per generated sequence

    if len(matched_pred) < 4:
        return {"spearman_rho_vs_truth": float("nan"), "n_matched": len(matched_pred)}

    rho, _ = spearmanr(matched_pred, matched_truth)
    return {
        "spearman_rho_vs_truth": float(rho),
        "n_matched":             len(matched_pred),
    }


# ---------------------------------------------------------------------------
# Chain mixing / MCMC diagnostics
# ---------------------------------------------------------------------------

def iact_1d(x: np.ndarray) -> float:
    """Integrated autocorrelation time using the initial monotone sequence estimator."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4:
        return 1.0
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-14:
        return 1.0
    acf = np.correlate(x, x, mode="full")[len(x) - 1:]
    acf = acf / acf[0]
    tau = 1.0
    for k in range(1, len(acf)):
        if acf[k] <= 0:
            break
        tau += 2.0 * acf[k]
    return float(max(tau, 1.0))


def chain_mixing_metrics(df: pd.DataFrame, runtime_sec: Optional[float]) -> Dict[str, float]:
    """
    Compute worst-chain IACT, minimum ESS, and ESS/sec.

    Reads 'chain' and 'score' columns; falls back to single-chain if absent.
    'runtime_sec' may come from the input CSV (column) or CLI flag.
    """
    if "score" not in df.columns:
        return {"iact": float("nan"), "ess": float("nan"), "ess_per_sec": float("nan")}

    if runtime_sec is None and "runtime_sec" in df.columns:
        runtime_sec = float(df["runtime_sec"].iloc[0])
    if runtime_sec is None or runtime_sec <= 0:
        runtime_sec = 1.0

    if "chain" in df.columns:
        groups = [g for _, g in df.groupby("chain")]
    else:
        groups = [df]

    taus, esss = [], []
    for g in groups:
        x = g.sort_values("step")["score"].to_numpy(dtype=np.float64) if "step" in g.columns else g["score"].to_numpy(dtype=np.float64)
        tau = iact_1d(x)
        taus.append(tau)
        esss.append(len(x) / tau)

    iact = float(np.max(taus))
    ess  = float(np.min(esss))
    return {
        "iact": iact,
        "ess": ess,
        "ess_per_sec": ess / runtime_sec,
    }


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------

def ensure_pdb(wt_pdb: str, results_dir: Path) -> Path:
    local = Path(wt_pdb)
    if local.exists():
        return local
    pdb_id = wt_pdb.upper().replace(".PDB", "")
    out = results_dir / f"{pdb_id}.pdb"
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", out)
    return out


def add_structure_metrics(
    df: pd.DataFrame,
    device: torch.device,
    wt_coords: np.ndarray,
    max_n: int,
) -> pd.DataFrame:
    try:
        import esm
        from tmtools import tm_align
        fold_model = esm.pretrained.esmfold_v1().eval().to(device)
        fold_model.set_chunk_size(128)
    except (ModuleNotFoundError, Exception):
        df["plddt_score"] = np.nan
        df["tm_score"] = np.nan
        df["rmsd"] = np.nan
        return df

    df = df.copy()
    df["plddt_score"] = np.nan
    df["tm_score"] = np.nan
    df["rmsd"] = np.nan

    score_col = "esm2_fitness" if "esm2_fitness" in df.columns else "pldm_regressor_score"
    order = df.sort_values(score_col, ascending=False).head(max_n).index

    with torch.no_grad():
        for idx in order:
            seq = str(df.loc[idx, "sequence"])
            try:
                pdb_txt = fold_model.infer_pdb(seq)
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
                    tm_score = float("nan")
                    rmsd = float("nan")
            except Exception:
                plddt = float("nan")
                tm_score = float("nan")
                rmsd = float("nan")
            df.loc[idx, "plddt_score"] = plddt
            df.loc[idx, "tm_score"] = tm_score
            df.loc[idx, "rmsd"] = rmsd

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir    = Path(__file__).resolve().parent
    proldm_root   = (script_dir / args.proldm_root).resolve()
    train_csv     = (proldm_root / args.train_csv).resolve()
    ae_ckpt       = (proldm_root / args.ae_ckpt).resolve()
    input_csv     = (script_dir / args.input_csv).resolve()
    results_csv   = (script_dir / args.results_csv).resolve()
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    for p in [train_csv, ae_ckpt, input_csv]:
        if not p.exists():
            raise FileNotFoundError(p)

    device    = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    train_df  = pd.read_csv(train_csv)
    wt_seq    = choose_wt(train_df)
    train_seq_col = infer_seq_col(train_df)
    train_fitness_col = infer_fitness_col(train_df)
    train_seqs = [clean_seq(str(s)) for s in train_df[train_seq_col].dropna().tolist()]

    sys.path.insert(0, str(proldm_root))
    from model.JTAE.models_condif_1d import jtae

    ckpt_raw = torch.load(ae_ckpt, map_location=device)
    state    = normalize_ckpt_state(ckpt_raw)
    dims     = infer_dims_from_state(state)
    hparams  = build_hparams(args.dataset, dims, device)
    model    = jtae(hparams).to(device)
    model.load_state_dict(filter_by_shape(state, model.state_dict()), strict=False)
    model.eval()

    raw     = pd.read_csv(input_csv)
    seq_col = infer_seq_col(raw)
    seqs    = [clean_seq(str(s)) for s in raw[seq_col].tolist()]

    # 1. Internal regressor score (batched)
    z = encode(model, seqs, dims["seq_len"], device, args.batch_size).to(device)
    with torch.no_grad():
        pldm_score = model.regressor_module(z).squeeze(-1).detach().cpu().numpy()

    # 2. External fitness score via ESM2 (optional)
    esm2 = ESM2Scorer(args.esm2_model, device, args.esm2_head_path) if args.use_esm2 else None
    if esm2 is not None:
        esm_scores, perplexities = [], []
        with torch.no_grad():
            for i in range(0, len(seqs), args.batch_size):
                chunk = seqs[i: i + args.batch_size]
                s, p = esm2.score_and_perplexity(chunk)
                esm_scores.extend(s.detach().cpu().tolist())
                perplexities.extend(p.detach().cpu().tolist())
    else:
        esm_scores    = [float("nan")] * len(seqs)
        perplexities  = [float("nan")] * len(seqs)

    # 3. Per-sequence novelty
    nearest_id, min_ham, nearest_hash = [], [], []
    for seq in seqs:
        bid, bh, bm = min_train_distance_metrics(seq, train_seqs)
        nearest_id.append(bid)
        min_ham.append(bh)
        nearest_hash.append(seq_hash(bm))

    out = pd.DataFrame({
        "sequence":               seqs,
        "id":                     [seq_hash(s) for s in seqs],
        "method_name":            args.method_name,
        "pldm_regressor_score":   pldm_score,
        "esm2_fitness":           esm_scores,
        "perplexity":             perplexities,
        "sequence_identity_wt":   [identity(s, wt_seq) for s in seqs],
        "hamming_dist_wt":        [hamming(s, wt_seq)  for s in seqs],
        "nearest_train_identity": nearest_id,
        "min_hamming_to_train":   min_ham,
        "nearest_train_hash":     nearest_hash,
    })

    # 4. Structure metrics (requires ESMFold + tmtools; skipped if absent)
    if args.with_structure:
        try:
            wt_pdb    = ensure_pdb(args.wt_pdb, results_csv.parent)
            wt_coords = parse_ca_coords_from_pdb(wt_pdb)
            out = add_structure_metrics(out, device, wt_coords, args.structure_max_per_method)
        except ModuleNotFoundError:
            out["plddt_score"] = float("nan")
            out["tm_score"]    = float("nan")
            out["rmsd"]        = float("nan")
    else:
        out["plddt_score"] = float("nan")
        out["tm_score"]    = float("nan")
        out["rmsd"]        = float("nan")

    out.to_csv(results_csv, index=False)
    print(f"Saved per-sequence results to: {results_csv}")

    # ------------------------------------------------------------------
    # 5. NEW -- Per-position KL divergence  D_KL(p_gen || p_train)
    # ------------------------------------------------------------------
    mean_kl, kl_per_pos = positional_kl_divergence(train_seqs, seqs)

    # Save per-position KL array as a companion file for inspection/plotting
    if len(kl_per_pos) > 0:
        kl_path = results_csv.with_suffix(".kl_per_pos.npy")
        np.save(kl_path, kl_per_pos)
        print(f"Saved per-position KL divergence array to: {kl_path}")

    # ------------------------------------------------------------------
    # 6. NEW -- LOO regressor calibration on training set
    # ------------------------------------------------------------------
    loo_metrics = loo_regressor_calibration(
        model, train_df, train_seq_col, train_fitness_col,
        dims["seq_len"], device, max_samples=args.loo_max_samples,
    )

    # ------------------------------------------------------------------
    # 7. NEW -- Spearman rho between regressor scores and true DMS labels
    # ------------------------------------------------------------------
    spearman_metrics = spearman_regressor_vs_truth(
        seqs, pldm_score, train_df, train_seq_col, train_fitness_col,
    )

    # 8. Set-level summary
    summary: Dict = {
        "method_name":                           args.method_name,
        "n_sequences":                           len(seqs),
        "mean_internal_regressor_score":         float(np.nanmean(pldm_score)),
        "mean_external_fitness_score":           float(np.nanmean(esm_scores)),
        "mean_nearest_train_identity":           float(np.nanmean(nearest_id)),
        "mean_min_hamming_to_train":             float(np.nanmean(min_ham)),
        "mean_identity_wt":                      float(np.nanmean(out["sequence_identity_wt"])),
        "entropy_mse_vs_train":                  entropy_mse(train_seqs, seqs),
        "pairwise_residue_correlation_vs_train": pairwise_residue_correlation(train_seqs, seqs),
        # NEW
        "mean_positional_kl_divergence":         mean_kl,
        "loo_spearman_rho":                      loo_metrics["loo_spearman_rho"],
        "loo_mse":                               loo_metrics["loo_mse"],
        "loo_mae":                               loo_metrics["loo_mae"],
        "loo_n":                                 loo_metrics["loo_n"],
        "spearman_rho_vs_truth":                 spearman_metrics["spearman_rho_vs_truth"],
        "n_matched_to_truth":                    spearman_metrics["n_matched"],
        # existing
        "mean_plddt_score":                      float(np.nanmean(out["plddt_score"])),
        "mean_rmsd":                             float(np.nanmean(out["rmsd"])),
    }

    mix = chain_mixing_metrics(raw, runtime_sec=args.runtime_sec)
    summary.update(mix)

    summary_path = results_csv.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved set-level summary to:   {summary_path}")

    print("\n=== Scorecard: {} ===".format(args.method_name))
    print(f"  Internal regressor score  : {summary['mean_internal_regressor_score']:.4f}")
    print(f"  External fitness (ESM2)   : {summary['mean_external_fitness_score']:.4f}")
    print(f"  Nearest-train identity    : {summary['mean_nearest_train_identity']:.4f}")
    print(f"  Min Hamming to train      : {summary['mean_min_hamming_to_train']:.1f}")
    print(f"  Entropy MSE vs train      : {summary['entropy_mse_vs_train']:.6f}")
    print(f"  Pairwise residue corr     : {summary['pairwise_residue_correlation_vs_train']:.4f}")
    print(f"  Mean pos. KL div (gen||tr): {summary['mean_positional_kl_divergence']:.6f}")
    print(f"  LOO Spearman rho          : {summary['loo_spearman_rho']:.4f}  (n={summary['loo_n']})")
    print(f"  LOO MSE / MAE             : {summary['loo_mse']:.4f} / {summary['loo_mae']:.4f}")
    print(f"  Spearman rho vs DMS truth : {summary['spearman_rho_vs_truth']:.4f}  (n_matched={summary['n_matched_to_truth']})")
    print(f"  Mean pLDDT                : {summary['mean_plddt_score']:.2f}")
    print(f"  Mean RMSD                 : {summary['mean_rmsd']:.2f}")
    print(f"  Worst-chain IACT          : {summary['iact']:.2f}")
    print(f"  Min-chain ESS             : {summary['ess']:.2f}")
    print(f"  ESS / sec                 : {summary['ess_per_sec']:.4f}")


if __name__ == "__main__":
    main()
