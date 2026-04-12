#!/usr/bin/env python3
"""
Experiment Runner: Gradient Ascent vs ESS vs TESS

Orchestrates the full optimization pipeline for all three methods:
1. Gradient Ascent  — classical gradient-based optimization
2. ESS              — Elliptical Slice Sampling (Murray et al. 2010)
3. TESS             — Transport ESS (Cabezas & Nemeth, AISTATS 2023)

Critical fix: training-prior sampling
-------------------------------------
The oracle was trained on a collapsed posterior: all training latents
lie in a tight cluster at x_mean ≈ [-0.83, 0.36, 0.59, 0.84, ...] with
x_std ≈ [0.08-0.29]. The model prior N(0,I) is 3-6σ away in every dim.

Using z ~ N(0,I) as start vectors causes:
  - Decoder gets OOD inputs  → degenerate PDB → Rg ≈ 3-5 Å always
  - Oracle predicts 50.6 Å at z=0 (should be ~26.8 Å = intercept)
  - Oracle range for N(0,I): 48 ± 74 Å → physically impossible values

Fix: always sample from N(x_mean, diag(x_std²)) — the training distribution.

Coordinate-frame fix (Rg pipeline)
-----------------------------------
The decoder outputs shape (1, 8, 3) — 8 skeleton anchor points in Å,
not a full residue chain.  gyr_pred.py is called AFTER decode and now
filters to CA-only atoms so that Rg is computed in the correct frame.

Gradient flow fix
-----------------
For gradient ascent, the score call must NOT be wrapped in torch.no_grad()
so that rg.backward() (or score.backward()) can propagate gradients back
through the oracle into z.

TESS double warm-start fix
--------------------------
run_tess() handles its own internal warm_start (gradient-ascent
pre-conditioning). experiment_runner must NOT run a separate GA warm-start
before calling run_tess — that would double-condition the starting point.

Temperature guidance
--------------------
With start vectors now in the correct region, score at z_start ≈ intercept.
Rule of thumb: T ≈ (intercept - target)^2 / 4
  target=5  Å → T ≈ (26.8-5)^2/4  ≈ 119  → use --temperature 120
  target=20 Å → T ≈ (26.8-20)^2/4 ≈  12  → use --temperature 12
  target=30 Å → T ≈ (30-26.8)^2/4 ≈   3  → use --temperature 3
Training-set Rg range (10th–90th pct): 14.7 Å – 72.2 Å
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import random


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run gradient ascent, ESS, and TESS optimization on random latent vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to ridge_latent_*_model.npz (trained oracle)",
    )
    parser.add_argument(
        "--target-tm",
        type=float,
        default=65.0,
        help="Target Rg value to optimize towards (Å). "
             "Training-set 10th–90th pct range: 14.7–72.2 Å.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=100.0,
        help=(
            "Likelihood temperature T: score = -(pred-target)^2 / T.  "
            "Rule of thumb: T ≈ (oracle_intercept - target)^2 / 4.  "
            "With intercept≈26.8 Å: target=5→T≈120, target=20→T≈12, "
            "target=30→T≈3.  Default 100.0 is safe for most Rg targets."
        ),
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=100,
        help="Number of random starting vectors",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of optimization steps per vector",
    )
    parser.add_argument(
        "--gradient-lr",
        type=float,
        default=0.1,
        help="Learning rate for gradient ascent",
    )
    parser.add_argument(
        "--gradient-penalty",
        type=float,
        default=0.05,
        help="L2 penalty weight for gradient ascent",
    )
    # ESS options
    parser.add_argument(
        "--ess-warm-start",
        action="store_true",
        default=True,
        help=(
            "Warm-start ESS/TESS: run 20 gradient-ascent steps before sampling. "
            "For TESS, warm_start is handled internally by run_tess(); "
            "this flag only affects the standalone ESS run."
        ),
    )
    parser.add_argument(
        "--ess-warm-steps",
        type=int,
        default=20,
        help="Number of gradient-ascent warm-start steps before ESS",
    )
    parser.add_argument(
        "--ess-warm-lr",
        type=float,
        default=0.1,
        help="Learning rate for ESS gradient-ascent warm-start",
    )
    # TESS options
    parser.add_argument(
        "--tess-n-transforms",
        type=int,
        default=2,
        help="Number of G-D coupling layer pairs in the normalizing flow",
    )
    parser.add_argument(
        "--tess-d-hidden",
        type=int,
        default=32,
        help="Hidden width of coupling MLP networks",
    )
    parser.add_argument(
        "--tess-warmup-epochs",
        type=int,
        default=3,
        help="Number of warm-up epochs h (Algorithm 2, Cabezas & Nemeth 2023)",
    )
    parser.add_argument(
        "--tess-warmup-chains",
        type=int,
        default=10,
        help="Chains k per warm-up epoch",
    )
    parser.add_argument(
        "--tess-m-grad-steps",
        type=int,
        default=10,
        help="Adam steps m per normalizing flow update",
    )
    parser.add_argument(
        "--tess-flow-lr",
        type=float,
        default=1e-3,
        help="Learning rate for normalizing flow Adam optimizer",
    )
    parser.add_argument(
        "--skip-tess",
        action="store_true",
        default=False,
        help="Skip TESS (run GA + ESS only)",
    )
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runner/results/optimization"),
        help="Directory to save z tensors and stats JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_start_vectors(
    n: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Sample start vectors from the training prior N(x_mean, diag(x_std²)).

    Using N(0,I) places vectors 3-6σ outside the oracle's training domain,
    causing degenerate decoder outputs and meaningless Rg values.
    """
    mean_t = torch.from_numpy(x_mean).float().to(device)
    std_t  = torch.from_numpy(x_std).float().to(device)
    eps    = torch.randn(n, len(x_mean), device=device)
    return mean_t.unsqueeze(0) + eps * std_t.unsqueeze(0)


# ---------------------------------------------------------------------------
# Gradient Ascent
# ---------------------------------------------------------------------------

def run_gradient_ascent(
    z_starts: torch.Tensor,
    target_fn,
    n_steps: int = 50,
    lr: float = 0.1,
    penalty: float = 0.05,
    x_mean: Optional[np.ndarray] = None,
    x_std:  Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Gradient ascent on the oracle log-likelihood.

    Gradient-flow fix
    -----------------
    The oracle call must NOT be inside torch.no_grad() — the gradient
    must flow back from log_likelihood through the oracle into z so
    that z.grad is populated on every step.

    Args:
        z_starts: Starting latent vectors [N, d]
        target_fn: TargetFunction (oracle)
        n_steps: Optimization steps
        lr: Adam learning rate
        penalty: L2 regularisation weight towards x_mean
        x_mean: Training-prior mean for L2 anchor (optional)
        x_std:  Training-prior std  (unused here, for API parity)

    Returns:
        z_final: Optimised latent vectors [N, d]
    """
    device   = z_starts.device
    z_final  = []
    mean_anchor = (
        torch.from_numpy(x_mean).float().to(device)
        if x_mean is not None else None
    )

    for i, z0 in enumerate(z_starts):
        z = z0.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)

        for step in range(n_steps):
            opt.zero_grad()
            # NOTE: no torch.no_grad() here — gradients must flow through oracle
            out   = target_fn(z)
            score = out.log_likelihood

            # L2 regularisation: penalise drift from training prior
            l2_penalty = torch.tensor(0.0, device=device)
            if mean_anchor is not None and penalty > 0.0:
                l2_penalty = penalty * ((z - mean_anchor) ** 2).sum()

            loss = -(score - l2_penalty)
            loss.backward()
            opt.step()

        z_final.append(z.detach())

        if (i + 1) % max(1, len(z_starts) // 10) == 0 or i == 0:
            final_score = target_fn(z.detach()).pred_value.item()
            print(
                f"  [GA] Vector {i+1:4d}/{len(z_starts)} | "
                f"pred={final_score:7.3f} | ||z||={z.detach().norm().item():.3f}"
            )

    return torch.stack(z_final)


# ---------------------------------------------------------------------------
# ESS
# ---------------------------------------------------------------------------

def run_ess(
    z_starts: torch.Tensor,
    target_fn,
    n_steps: int = 50,
    warm_start: bool = True,
    warm_steps: int = 20,
    warm_lr: float = 0.1,
    x_mean: Optional[np.ndarray] = None,
    x_std:  Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Elliptical Slice Sampling (Murray, Adams & MacKay 2010).

    Training-prior fix: auxiliary variable v is drawn from N(x_mean, diag(x_std²))
    so that ellipse proposals stay in the decoder's valid input domain.
    """
    import math

    device  = z_starts.device
    z_final = []
    mean_t  = torch.from_numpy(x_mean).float().to(device) if x_mean is not None else None
    std_t   = torch.from_numpy(x_std).float().to(device)  if x_std  is not None else None

    def log_score(z: torch.Tensor) -> float:
        with torch.no_grad():
            return target_fn(z).log_likelihood.item()

    def _ess_step(z: torch.Tensor) -> torch.Tensor:
        current = log_score(z)
        log_s   = current + math.log(random.uniform(0, 1) + 1e-300)

        eps = torch.randn_like(z)
        v   = (mean_t + eps * std_t) if (mean_t is not None) else eps

        theta     = random.uniform(0, 2 * math.pi)
        theta_min = theta - 2 * math.pi
        theta_max = theta

        for _ in range(100):
            z_prime   = z * math.cos(theta) + v * math.sin(theta)
            if log_score(z_prime) > log_s:
                return z_prime
            if theta > 0:
                theta_max = theta
            else:
                theta_min = theta
            theta = random.uniform(theta_min, theta_max)
        return z

    for i, z0 in enumerate(z_starts):
        z = z0.clone().detach()

        # Optional gradient-ascent warm-start
        if warm_start:
            z_ws = z.requires_grad_(True)
            opt  = torch.optim.Adam([z_ws], lr=warm_lr)
            for _ in range(warm_steps):
                opt.zero_grad()
                (-target_fn(z_ws).log_likelihood).backward()
                opt.step()
            z = z_ws.detach()

        for _ in range(n_steps):
            z = _ess_step(z)

        z_final.append(z)

        if (i + 1) % max(1, len(z_starts) // 10) == 0 or i == 0:
            print(
                f"  [ESS] Vector {i+1:4d}/{len(z_starts)} | "
                f"pred={target_fn(z).pred_value.item():7.3f} | ||z||={z.norm().item():.3f}"
            )

    return torch.stack(z_final)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load oracle
    from optimization.target_function import TargetFunction
    target_fn = TargetFunction.from_checkpoint(
        model_path=args.checkpoint,
        target_value=args.target_tm,
        temperature=args.temperature,
    )
    target_fn = target_fn.to(device)
    target_fn.eval()

    # Extract training-prior statistics from oracle checkpoint
    with np.load(args.checkpoint) as npz:
        x_mean = npz["x_mean"].astype(np.float32)
        x_std  = npz["x_std"].astype(np.float32)
        intercept = float(npz["intercept"].flat[0])

    print(f"Oracle intercept (pred at x_mean): {intercept:.2f} Å")
    print(f"Training-set Rg range (10–90 pct): 14.7 – 72.2 Å")
    print(f"Target Rg: {args.target_tm:.1f} Å")

    # Sample starting vectors from training prior (not N(0,I))
    z_starts = sample_start_vectors(args.n_vectors, x_mean, x_std, device)
    print(f"Sampled {args.n_vectors} start vectors from N(x_mean, diag(x_std²))")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"target_tm": args.target_tm, "temperature": args.temperature}

    # ── Gradient Ascent ──────────────────────────────────────────────────
    print("\n[1/3] Gradient Ascent")
    z_ga = run_gradient_ascent(
        z_starts,
        target_fn,
        n_steps=args.n_steps,
        lr=args.gradient_lr,
        penalty=args.gradient_penalty,
        x_mean=x_mean,
        x_std=x_std,
    )
    torch.save(z_ga, args.output_dir / "z_final_gradient_ascent.pt")

    with torch.no_grad():
        preds_ga = target_fn(z_ga).pred_value.cpu().numpy()
    stats["gradient_ascent"] = {
        "mean": float(preds_ga.mean()),
        "max":  float(preds_ga.max()),
        "std":  float(preds_ga.std()),
        "improvement_mean": float((preds_ga - intercept).mean()),
    }
    print(f"  GA   pred: mean={preds_ga.mean():.2f}  max={preds_ga.max():.2f}  std={preds_ga.std():.2f}")

    # ── ESS ──────────────────────────────────────────────────────────────
    print("\n[2/3] ESS")
    z_ess = run_ess(
        z_starts,
        target_fn,
        n_steps=args.n_steps,
        warm_start=args.ess_warm_start,
        warm_steps=args.ess_warm_steps,
        warm_lr=args.ess_warm_lr,
        x_mean=x_mean,
        x_std=x_std,
    )
    torch.save(z_ess, args.output_dir / "z_final_ess.pt")

    with torch.no_grad():
        preds_ess = target_fn(z_ess).pred_value.cpu().numpy()
    stats["ess"] = {
        "mean": float(preds_ess.mean()),
        "max":  float(preds_ess.max()),
        "std":  float(preds_ess.std()),
        "improvement_mean": float((preds_ess - intercept).mean()),
    }
    print(f"  ESS  pred: mean={preds_ess.mean():.2f}  max={preds_ess.max():.2f}  std={preds_ess.std():.2f}")

    # ── TESS ─────────────────────────────────────────────────────────────
    if not args.skip_tess:
        print("\n[3/3] TESS")
        from optimization.transport_ess import run_tess

        x_mean_t = torch.from_numpy(x_mean).to(device)
        x_std_t  = torch.from_numpy(x_std).to(device)

        z_tess_list = []
        for i, z0 in enumerate(z_starts):
            # NOTE: do NOT run a GA warm-start here.
            # run_tess() calls its own internal warm_start when warm_start=True.
            # Double warm-starting corrupts the starting point.
            z_final_i, _ = run_tess(
                z_start=z0,
                target_fn=target_fn,
                steps=args.n_steps,
                n_transforms=args.tess_n_transforms,
                d_hidden=args.tess_d_hidden,
                warmup_epochs=args.tess_warmup_epochs,
                warmup_chains=args.tess_warmup_chains,
                m_grad_steps=args.tess_m_grad_steps,
                flow_lr=args.tess_flow_lr,
                warm_start=args.ess_warm_start,
                warm_steps=args.ess_warm_steps,
                warm_lr=args.ess_warm_lr,
                x_mean=x_mean_t,
                x_std=x_std_t,
            )
            z_tess_list.append(z_final_i)

            if (i + 1) % max(1, args.n_vectors // 10) == 0 or i == 0:
                pred_i = target_fn(z_final_i).pred_value.item()
                print(
                    f"  [TESS] Vector {i+1:4d}/{args.n_vectors} | "
                    f"pred={pred_i:7.3f} | ||z||={z_final_i.norm().item():.3f}"
                )

        z_tess = torch.stack(z_tess_list)
        torch.save(z_tess, args.output_dir / "z_final_tess.pt")

        with torch.no_grad():
            preds_tess = target_fn(z_tess).pred_value.cpu().numpy()
        stats["tess"] = {
            "mean": float(preds_tess.mean()),
            "max":  float(preds_tess.max()),
            "std":  float(preds_tess.std()),
            "improvement_mean": float((preds_tess - intercept).mean()),
        }
        print(f"  TESS pred: mean={preds_tess.mean():.2f}  max={preds_tess.max():.2f}  std={preds_tess.std():.2f}")
    else:
        print("\n[3/3] TESS — skipped (--skip-tess)")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Target Rg:  {args.target_tm:.1f} Å")
    methods = [k for k in ("gradient_ascent", "ess", "tess") if k in stats]
    for m in methods:
        s = stats[m]
        print(
            f"  {m:17s}: mean={s['mean']:7.2f}  max={s['max']:7.2f}  "
            f"std={s['std']:6.2f}  Δmean={s['improvement_mean']:+.2f}"
        )
    if methods:
        winner = max(methods, key=lambda m: stats[m]["max"])
        print(f"  Winner (by max pred): {winner}")
    print("=" * 60)

    stats_path = args.output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved → {stats_path}")


if __name__ == "__main__":
    main()
