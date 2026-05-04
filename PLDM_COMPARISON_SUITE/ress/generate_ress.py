"""
generate_ress.py  --  Restart ESS (RESS) sequence generator

Algorithm
---------
Round 0:
  Build num_chains starting latents from the top-K diverse training sequences
  (ranked by fitness, filtered so pairwise Hamming >= min_hamming_diversity).
  Run each chain for max_steps steps; collect unique (seq, latent, score).

Round r (r = 1 .. ress_rounds):
  1. From the global accepted pool, pick top-K latents by score, subject to
     the pairwise Hamming diversity filter on their *decoded sequences*.
  2. Widen temperature: T_r = T_base * (restart_temp_scale ** r)
  3. Run num_chains chains, distributing the K seeds round-robin.
  4. Accumulate any newly seen sequences into the global pool.
  5. Early-stop if new_unique_this_round < patience.

Output:
  Top-N sequences by score from the global pool (fully deduplicated).
  Also writes a per-round log CSV.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

# ---- resolve repo paths so we can import from PLDM_COMPARISON_SUITE --------
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../PLDM_COMPARISON_SUITE/ress
_SUITE_DIR  = _SCRIPT_DIR.parent                       # .../PLDM_COMPARISON_SUITE
sys.path.insert(0, str(_SUITE_DIR))

from pipeline_utils import (
    ESM2Scorer,
    build_hparams,
    choose_wt,
    clean_seq,
    filter_by_shape,
    infer_dims_from_state,
    infer_fitness_col,
    infer_seq_col,
    inds_to_seq,
    normalize_ckpt_state,
    seq_to_inds,
)
from sampler_ess import RunningNorm, TransportESSSampler, ess_step


# ===========================================================================
#  CLI
# ===========================================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"1", "true", "yes", "y", "t"}:
        return True
    if v.lower() in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RESS: Restart Elliptical Slice Sampling")
    # Paths
    p.add_argument("--proldm-root",  default="../PROLDM_OUTLIER")
    p.add_argument("--train-csv",    default="data/mut_data/GFP-train.csv")
    p.add_argument("--ae-ckpt",      default="train_logs/GFP/epoch_1000.pt")
    p.add_argument("--dataset",      default="GFP")
    p.add_argument("--out-dir",      default="ress/outputs/GFP")
    # Generation
    p.add_argument("--n",            type=int,   default=1000)
    p.add_argument("--num-chains",   type=int,   default=8)
    p.add_argument("--burnin",       type=int,   default=100)
    p.add_argument("--max-steps",    type=int,   default=8000)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default="cpu")
    # Likelihood
    p.add_argument("--alpha",        type=float, default=0.0,
                   help="ESM2 weight in likelihood (0 = regressor only)")
    p.add_argument("--esm2-model",   default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--esm2-head-path", default=None)
    p.add_argument("--use-esm2",     type=str2bool, default=False)
    # ESS core
    p.add_argument("--latent-temperature", type=float, default=1.0,
                   help="Base ESS temperature T (1.0 = exact posterior)")
    p.add_argument("--tess-delta",   type=float, default=0.5,
                   help="Ball radius around seed latent. Use 0 to disable.")
    # RESS restart
    p.add_argument("--ress-rounds",       type=int,   default=5)
    p.add_argument("--ress-top-k",        type=int,   default=8)
    p.add_argument("--ress-min-hamming",  type=int,   default=3,
                   help="Min AA Hamming distance between restart seeds (diversity filter)")
    p.add_argument("--ress-patience",     type=int,   default=50,
                   help="Stop early if fewer than this many new unique seqs found in a round")
    p.add_argument("--ress-temp-scale",   type=float, default=1.05,
                   help="Multiply temperature by this factor each restart round")
    # Transport (optional)
    p.add_argument("--use-transport",     type=str2bool, default=False)
    p.add_argument("--flow-buffer-size",  type=int,   default=128)
    p.add_argument("--flow-adapt-every",  type=int,   default=32)
    p.add_argument("--flow-lr",           type=float, default=1e-3)
    p.add_argument("--flow-adapt-steps",  type=int,   default=5)
    return p.parse_args()


# ===========================================================================
#  Likelihood  (same EMA-normalised wrapper as generate_sequences.py)
# ===========================================================================

class Likelihood:
    def __init__(self, ae, esm2, seq_len, device, alpha=0.0, ema_alpha=0.05):
        self.ae      = ae
        self.esm2    = esm2
        self.seq_len = seq_len
        self.device  = device
        self.alpha   = alpha
        self._norm_reg = RunningNorm(alpha=ema_alpha)
        self._norm_esm = RunningNorm(alpha=ema_alpha)

    @torch.no_grad()
    def decode_seq(self, z: torch.Tensor) -> str:
        z_ = z if z.dim() == 2 else z.unsqueeze(0)
        logits = self.ae.decode(z_)
        idx = torch.argmax(logits, dim=1).squeeze(0)
        return clean_seq(inds_to_seq(idx))

    @torch.no_grad()
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        z_ = z if z.dim() == 2 else z.unsqueeze(0)
        reg = self.ae.regressor_module(z_).squeeze()
        reg_n = self._norm_reg.update_and_normalize(reg)
        if self.esm2 is None or self.alpha == 0.0:
            return reg_n
        seq = self.decode_seq(z_)
        esm, _ = self.esm2.score_and_perplexity([seq])
        esm_n = self._norm_esm.update_and_normalize(esm.squeeze())
        return self.alpha * esm_n + (1.0 - self.alpha) * reg_n


# ===========================================================================
#  Helpers
# ===========================================================================

def hamming(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:n], b[:n])) + abs(len(a) - len(b))


def diverse_topk(
    pool: List[Dict],
    k: int,
    min_hamming: int,
) -> List[Dict]:
    """
    Greedy diverse top-K selection.
    Sort pool by score descending; greedily pick entries whose decoded
    sequence is at least min_hamming away from all already-selected entries.
    """
    sorted_pool = sorted(pool, key=lambda r: r["score"], reverse=True)
    selected: List[Dict] = []
    for entry in sorted_pool:
        if len(selected) >= k:
            break
        seq = entry["sequence"]
        if all(hamming(seq, s["sequence"]) >= min_hamming for s in selected):
            selected.append(entry)
    # If diversity filter was too strict, top up with best remaining
    if len(selected) < k:
        used_seqs = {s["sequence"] for s in selected}
        for entry in sorted_pool:
            if len(selected) >= k:
                break
            if entry["sequence"] not in used_seqs:
                selected.append(entry)
                used_seqs.add(entry["sequence"])
    return selected


@torch.no_grad()
def _ess_step(
    z: torch.Tensor,
    ll_current: torch.Tensor,
    ll_fn,
    temperature: float,
    delta: Optional[float],
    center: Optional[torch.Tensor],
    max_attempts: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Single ESS step with optional ball constraint."""
    nu      = temperature * torch.randn_like(z)
    log_y   = ll_current + torch.log(torch.rand(1, device=z.device))
    theta   = torch.rand(1, device=z.device) * 2.0 * math.pi
    t_min   = theta - 2.0 * math.pi
    t_max   = theta.clone()

    for _ in range(max_attempts):
        prop = z * torch.cos(theta) + nu * torch.sin(theta)
        # optional ball constraint
        if center is not None and delta is not None and delta > 0:
            if torch.norm(prop - center).item() > delta:
                if theta.item() < 0:
                    t_min = theta
                else:
                    t_max = theta
                theta = torch.empty(1, device=z.device).uniform_(t_min.item(), t_max.item())
                continue
        ll_prop = ll_fn(prop)
        if ll_prop > log_y:
            return prop, ll_prop, True
        if theta.item() < 0:
            t_min = theta
        else:
            t_max = theta
        theta = torch.empty(1, device=z.device).uniform_(t_min.item(), t_max.item())

    return z, ll_current, False


@torch.no_grad()
def run_chains(
    ae,
    ll_fn: Likelihood,
    z_seeds: List[torch.Tensor],
    num_chains: int,
    burnin: int,
    max_steps: int,
    temperature: float,
    delta: float,
    use_transport: bool,
    flow_buffer_size: int,
    flow_adapt_every: int,
    flow_lr: float,
    flow_adapt_steps: int,
    latent_dim: int,
    device: torch.device,
    round_idx: int,
    global_seen: set,
) -> Tuple[List[Dict], int]:
    """
    Run num_chains ESS/TESS chains, each seeded from z_seeds[chain % len(z_seeds)].
    Returns:
        new_rows  -- list of dicts for newly seen sequences only
        n_new     -- number of new unique sequences found this round
    """
    new_rows: List[Dict] = []

    for chain_idx in range(num_chains):
        z = z_seeds[chain_idx % len(z_seeds)].clone().to(device)
        ll = ll_fn(z)
        center = z.clone()   # ball centered on this chain's seed

        if use_transport:
            sampler = TransportESSSampler(
                z_init=z,
                log_likelihood_fn=ll_fn,
                latent_dim=latent_dim,
                temperature=temperature,
                buffer_size=flow_buffer_size,
                adapt_every=flow_adapt_every,
                flow_lr=flow_lr,
                n_adapt_steps=flow_adapt_steps,
                device=device,
            )

        for step in range(max_steps):
            if use_transport:
                z, accepted = sampler.step()
                ll = ll_fn(z)
            else:
                z, ll, accepted = _ess_step(
                    z, ll, ll_fn,
                    temperature=temperature,
                    delta=delta if delta > 0 else None,
                    center=center if delta > 0 else None,
                )

            if step < burnin:
                continue

            seq = ll_fn.decode_seq(z)
            if seq in global_seen:
                continue

            global_seen.add(seq)
            new_rows.append({
                "sequence":         seq,
                "score":            float(ll.item() if hasattr(ll, "item") else ll),
                "chain":            chain_idx,
                "step":             step,
                "round":            round_idx,
                "latent":           z.detach().cpu(),   # kept for restart seeding
            })

    return new_rows, len(new_rows)


# ===========================================================================
#  RESS main loop
# ===========================================================================

def ress(
    ae,
    ll_fn: Likelihood,
    train_df: pd.DataFrame,
    seq_col: str,
    fit_col: Optional[str],
    n: int,
    num_chains: int,
    burnin: int,
    max_steps: int,
    base_temperature: float,
    delta: float,
    ress_rounds: int,
    ress_top_k: int,
    ress_min_hamming: int,
    ress_patience: int,
    ress_temp_scale: float,
    use_transport: bool,
    flow_buffer_size: int,
    flow_adapt_every: int,
    flow_lr: float,
    flow_adapt_steps: int,
    latent_dim: int,
    device: torch.device,
    seq_len: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (results_df, round_log_df)
    """
    global_seen: set = set()
    global_pool: List[Dict] = []   # all accepted unique entries across rounds
    round_log: List[Dict] = []
    t0 = time.perf_counter()

    # ---- Round 0 seeds: top-K diverse training sequences ------------------
    if fit_col is not None:
        ranked = train_df.dropna(subset=[fit_col]).sort_values(fit_col, ascending=False)
    else:
        ranked = train_df.copy()
    ranked = ranked.drop_duplicates(subset=[seq_col])

    # Build initial z_seeds from top training seqs, diversity-filtered
    train_seed_rows: List[Dict] = []
    for _, row in ranked.iterrows():
        seq = clean_seq(str(row[seq_col]))
        if all(hamming(seq, s["sequence"]) >= ress_min_hamming for s in train_seed_rows):
            z = ae.encode(
                seq_to_inds(seq, seq_len).unsqueeze(0).to(device)
            ).squeeze(0).to(torch.float32)
            fitness = float(row[fit_col]) if fit_col else 0.0
            train_seed_rows.append({"sequence": seq, "score": fitness, "latent": z.cpu()})
        if len(train_seed_rows) >= ress_top_k:
            break

    if not train_seed_rows:
        raise RuntimeError("Could not build diverse training seeds. Check min_hamming or train CSV.")

    print(f"[ress] Round 0 seeds: {len(train_seed_rows)} diverse training sequences")

    temperature = base_temperature

    for round_idx in range(ress_rounds + 1):   # round 0 + ress_rounds restarts
        # Select seeds for this round
        if round_idx == 0:
            seed_entries = train_seed_rows
        else:
            # top-K diverse from global pool
            seed_entries = diverse_topk(global_pool, k=ress_top_k, min_hamming=ress_min_hamming)
            if not seed_entries:
                print(f"[ress] Round {round_idx}: pool empty — stopping")
                break
            temperature = base_temperature * (ress_temp_scale ** round_idx)
            print(f"[ress] Round {round_idx}: {len(seed_entries)} seeds, T={temperature:.4f}")

        z_seeds = [e["latent"].to(device) for e in seed_entries]

        new_rows, n_new = run_chains(
            ae=ae,
            ll_fn=ll_fn,
            z_seeds=z_seeds,
            num_chains=num_chains,
            burnin=burnin,
            max_steps=max_steps,
            temperature=temperature,
            delta=delta,
            use_transport=use_transport,
            flow_buffer_size=flow_buffer_size,
            flow_adapt_every=flow_adapt_every,
            flow_lr=flow_lr,
            flow_adapt_steps=flow_adapt_steps,
            latent_dim=latent_dim,
            device=device,
            round_idx=round_idx,
            global_seen=global_seen,
        )

        global_pool.extend(new_rows)
        best_score = max((r["score"] for r in global_pool), default=float("nan"))
        elapsed = time.perf_counter() - t0

        print(
            f"[ress] Round {round_idx}: +{n_new} new seqs "
            f"(total {len(global_pool)}) | best_score={best_score:.4f} | "
            f"T={temperature:.4f} | elapsed={elapsed:.1f}s"
        )
        round_log.append({
            "round":     round_idx,
            "n_new":     n_new,
            "n_total":   len(global_pool),
            "best_score": best_score,
            "temperature": temperature,
            "elapsed_sec": elapsed,
        })

        # Early stop
        if round_idx > 0 and n_new < ress_patience:
            print(f"[ress] Early stop: only {n_new} new seqs (< patience={ress_patience})")
            break

        # Already have enough unique sequences
        if len(global_pool) >= n:
            print(f"[ress] Collected {len(global_pool)} >= {n} requested — stopping")
            break

    # ---- Build output dataframes ------------------------------------------
    # Sort global pool by score descending, take top-n
    global_pool_sorted = sorted(global_pool, key=lambda r: r["score"], reverse=True)
    top_n = global_pool_sorted[:n]

    results_rows = []
    for r in top_n:
        results_rows.append({
            "sequence": r["sequence"],
            "score":    r["score"],
            "chain":    r["chain"],
            "step":     r["step"],
            "round":    r["round"],
        })

    results_df  = pd.DataFrame(results_rows)
    round_log_df = pd.DataFrame(round_log)
    return results_df, round_log_df


# ===========================================================================
#  Entry point
# ===========================================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    proldm_root = (_SUITE_DIR / args.proldm_root).resolve()
    train_csv   = (proldm_root / args.train_csv).resolve()
    ae_ckpt     = (proldm_root / args.ae_ckpt).resolve()
    out_dir     = (_SUITE_DIR / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [proldm_root, train_csv, ae_ckpt]:
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )

    train_df = pd.read_csv(train_csv)
    seq_col  = infer_seq_col(train_df)
    fit_col  = infer_fitness_col(train_df)

    # Load AE
    sys.path.insert(0, str(proldm_root))
    from model.JTAE.models_condif_1d import jtae

    ckpt_raw  = torch.load(ae_ckpt, map_location=device)
    ae_state  = normalize_ckpt_state(ckpt_raw)
    dims      = infer_dims_from_state(ae_state)
    hparams   = build_hparams(args.dataset, dims, device, batch_size=16)
    ae        = jtae(hparams).to(device)
    ae.load_state_dict(filter_by_shape(ae_state, ae.state_dict()), strict=False)
    ae.eval()

    esm2 = ESM2Scorer(args.esm2_model, device=device, head_path=args.esm2_head_path) \
           if args.use_esm2 else None

    ll_fn = Likelihood(
        ae=ae, esm2=esm2,
        seq_len=dims["seq_len"],
        device=device,
        alpha=args.alpha,
    )

    mode_str = "Transport ESS" if args.use_transport else "ESS"
    print(f"[ress] Starting RESS ({mode_str} per chain)")
    print(f"[ress] rounds={args.ress_rounds}, top_k={args.ress_top_k}, "
          f"min_hamming={args.ress_min_hamming}, patience={args.ress_patience}, "
          f"temp_scale={args.ress_temp_scale}, base_T={args.latent_temperature}")

    results_df, round_log_df = ress(
        ae=ae,
        ll_fn=ll_fn,
        train_df=train_df,
        seq_col=seq_col,
        fit_col=fit_col,
        n=args.n,
        num_chains=args.num_chains,
        burnin=args.burnin,
        max_steps=args.max_steps,
        base_temperature=args.latent_temperature,
        delta=args.tess_delta,
        ress_rounds=args.ress_rounds,
        ress_top_k=args.ress_top_k,
        ress_min_hamming=args.ress_min_hamming,
        ress_patience=args.ress_patience,
        ress_temp_scale=args.ress_temp_scale,
        use_transport=args.use_transport,
        flow_buffer_size=args.flow_buffer_size,
        flow_adapt_every=args.flow_adapt_every,
        flow_lr=args.flow_lr,
        flow_adapt_steps=args.flow_adapt_steps,
        latent_dim=dims.get("latent_dim", 32),
        device=device,
        seq_len=dims["seq_len"],
    )

    method_tag = "ress" if not args.use_transport else "ress_transport"
    results_df["method_name"] = method_tag
    results_df.to_csv(out_dir / f"results_{method_tag}.csv", index=False)
    round_log_df.to_csv(out_dir / "ress_round_log.csv", index=False)

    print(f"[ress] Saved {len(results_df)} sequences to {out_dir}/results_{method_tag}.csv")
    print(f"[ress] Round log saved to {out_dir}/ress_round_log.csv")

    # Print round summary
    print("\n=== RESS Round Summary ===")
    print(round_log_df.to_string(index=False))


if __name__ == "__main__":
    main()
