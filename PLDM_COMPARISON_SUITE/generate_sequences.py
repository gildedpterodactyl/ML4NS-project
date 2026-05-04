import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

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
    p = argparse.ArgumentParser(description="Generate baseline PLDM + ESS + TESS sequence sets.")
    p.add_argument("--proldm-root", type=str, default="../PROLDM_OUTLIER")
    p.add_argument("--train-csv", type=str, default="data/mut_data/GFP-train.csv")
    p.add_argument("--baseline-ckpt", type=str, default="train_logs/GFP/dropout_tiny_epoch_1000.pt")
    p.add_argument("--ae-ckpt", type=str, default="train_logs/GFP/epoch_1000.pt")
    p.add_argument("--dataset", type=str, default="GFP")
    p.add_argument("--esm2-model", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--esm2-head-path", type=str, default=None)
    p.add_argument("--use-esm2", type=str2bool, default=False)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--num-chains", type=int, default=8)
    # ------------------------------------------------------------------ #
    # ESS / TESS core hyperparameters                                      #
    #                                                                      #
    # tess-delta (ball radius)                                             #
    #   Raised from 0.14 → 0.5.  Latent vectors have typical L2 norm      #
    #   ~sqrt(latent_dim) ≈ 5-6 for dim=32, so 0.14 covered only ~0.6%   #
    #   of that norm -- the constraint fired on nearly every step and       #
    #   collapsed the angle bracket in 3-4 iterations, starving chains     #
    #   of unique proposals.  0.5 gives ~8-9% of the norm: meaningfully   #
    #   constraining without killing exploration.                           #
    #                                                                      #
    # delta-final                                                          #
    #   Optional linear anneal target for the ball radius.  Setting to    #
    #   1.0 loosens the ball as the chain warms up (step 0→max_steps maps  #
    #   delta→delta_final), allowing late-chain exploration.               #
    #                                                                      #
    # latent-temperature                                                   #
    #   Raised from 0.75 → 1.2.  T<1 compresses ν ~ N(0, T²I) toward 0,  #
    #   keeping proposals close to z_current; T=1 is the exact Gaussian   #
    #   prior; T=1.2 widens the ellipse so chains can jump across shallow  #
    #   valleys and reach higher-fitness peaks.  The log-likelihood        #
    #   threshold in the accept/reject step still enforces correctness     #
    #   (bracket shrinks until a valid point is found), so higher T only   #
    #   increases *proposal range*, not raw accept rate.                   #
    # ------------------------------------------------------------------ #
    p.add_argument("--tess-delta", type=float, default=0.5,
                   help="Ball radius constraining proposals around seed latent. "
                        "Raised from 0.14 to 0.5 -- see comment in parse_args.")
    p.add_argument("--delta-final", type=float, default=1.0,
                   help="If set, linearly anneal delta from tess-delta → delta-final "
                        "over the chain.  1.0 = gradually release the ball constraint.")
    p.add_argument("--latent-temperature", type=float, default=1.2,
                   help="ESS temperature T: scales auxiliary ν ~ N(0, T²I). "
                        "Raised from 0.75 to 1.2 to widen ellipse and improve mixing.")
    p.add_argument("--use-transport", type=str2bool, default=False)
    p.add_argument("--transport-strength", type=float, default=0.5)
    # ------------------------------------------------------------------ #
    # Burn-in / steps                                                      #
    #                                                                      #
    # burnin: 20 → 100                                                     #
    #   Short burnin meant samples were collected before the EMA           #
    #   normaliser had accumulated enough statistics, biasing early        #
    #   score estimates.  100 steps gives the RunningNorm ~5 half-lives    #
    #   at the default ema_alpha=0.05.                                     #
    #                                                                      #
    # max-steps: 5000 → 10000                                              #
    #   More steps directly increases unique-sequence yield before the     #
    #   bracket collapses or the per-chain quota is met.                   #
    # ------------------------------------------------------------------ #
    p.add_argument("--burnin", type=int, default=100,
                   help="Steps to discard before collecting samples. Raised 20 → 100.")
    p.add_argument("--max-steps", type=int, default=10000,
                   help="Max MCMC steps per chain. Raised 5000 → 10000.")
    p.add_argument("--mode", type=str, choices=["baseline", "ess", "tess", "transport_ess", "all"], default="all")
    p.add_argument("--omega", type=float, default=20.0)
    p.add_argument("--esm-weight", type=float, default=0.5)
    p.add_argument("--reg-weight", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--model-type", type=str, choices=["auto", "tiny", "standard"], default="auto")
    p.add_argument("--baseline-model-type", type=str, choices=["auto", "tiny", "standard"], default="auto")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs")
    # TESS flow hyperparameters
    p.add_argument("--flow-buffer-size", type=int, default=128)
    p.add_argument("--flow-adapt-every", type=int, default=32)
    p.add_argument("--flow-lr", type=float, default=1e-3)
    p.add_argument("--flow-adapt-steps", type=int, default=5)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def diffusion_baseline(
    proldm_root: Path,
    ckpt_path: Path,
    dataset: str,
    n: int,
    label: int,
    omega: float,
    device: torch.device,
) -> List[str]:
    sys.path.insert(0, str(proldm_root))
    from model.JTAE.models_condif_1d import jtae
    from model.ConDiff.DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionSampler

    ckpt = torch.load(ckpt_path, map_location=device)
    dims = infer_dims_from_state(ckpt)
    hparams = build_hparams(dataset, dims, device, batch_size=max(8, min(256, n)))
    hparams.dif_w = float(omega)

    model = jtae(hparams).to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    labels = (label * torch.ones(n)).long().to(device)
    sampler = GaussianDiffusionSampler(
        model.diff_model,
        hparams.dif_beta_1,
        hparams.dif_beta_T,
        hparams.dif_T,
        hparams,
    ).to(device)

    noisy_z = torch.randn(size=[n, 1, hparams.latent_dim], device=device)
    sampled_z = sampler(noisy_z, labels)
    sampled_z = sampled_z.squeeze(1).to(torch.float32)
    x_hat = model.decode(sampled_z).argmax(dim=1)
    return [clean_seq(inds_to_seq(row)) for row in x_hat]


class Likelihood:
    """
    Combined fitness log-likelihood for ESS/TESS.

    Uses RunningNorm (EMA) to normalise each score stream so single-sample
    evaluation (batch size = 1, common in MCMC steps) is numerically stable
    -- avoids the std=0 divide-by-zero present in the previous batch-norm impl.
    """

    def __init__(
        self,
        ae: torch.nn.Module,
        esm2: ESM2Scorer,
        seq_len: int,
        device: torch.device,
        esm_w: float,
        reg_w: float,
        alpha: float,
        ema_alpha: float = 0.05,
    ):
        self.ae = ae
        self.esm2 = esm2
        self.seq_len = seq_len
        self.device = device
        self.esm_w = esm_w
        self.reg_w = reg_w
        self.alpha = alpha

        self._norm_reg = RunningNorm(alpha=ema_alpha)
        self._norm_esm = RunningNorm(alpha=ema_alpha)

    @torch.no_grad()
    def decode_seq(self, z: torch.Tensor) -> List[str]:
        logits = self.ae.decode(z)
        idx = torch.argmax(logits, dim=1)
        return [clean_seq(inds_to_seq(x)) for x in idx]

    @torch.no_grad()
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        # z may be (D,) or (1, D); normalise to (1, D) for regressor
        z_ = z if z.dim() == 2 else z.unsqueeze(0)
        reg = self.ae.regressor_module(z_).squeeze(-1)  # (B,) or scalar

        # Reduce to scalar for single-sample MCMC steps
        reg_scalar = reg.squeeze()
        reg_n = self._norm_reg.update_and_normalize(reg_scalar)

        if self.esm2 is None:
            return reg_n

        seqs = self.decode_seq(z_)
        esm, _ = self.esm2.score_and_perplexity(seqs)
        esm_scalar = esm.squeeze()
        esm_n = self._norm_esm.update_and_normalize(esm_scalar)

        if self.alpha is not None:
            return self.alpha * esm_n + (1.0 - self.alpha) * reg_n
        return self.esm_w * esm_n + self.reg_w * reg_n


def infer_checkpoint_model_type(ckpt_path: Path) -> str:
    name = ckpt_path.name.lower()
    if "tiny" in name:
        return "tiny"
    return "standard"


def validate_model_type(model_type: str, ckpt_path: Path, label: str) -> None:
    inferred = infer_checkpoint_model_type(ckpt_path)
    if model_type == "auto":
        print(f"[model-check] {label}: inferred '{inferred}' from checkpoint {ckpt_path.name}")
        return
    if model_type != inferred:
        raise ValueError(
            f"[model-check] {label}: args.model_type='{model_type}' does not match checkpoint "
            f"'{ckpt_path.name}' (inferred '{inferred}')."
        )
    print(f"[model-check] {label}: args.model_type='{model_type}' matches checkpoint {ckpt_path.name}")


def build_chain_starts(
    train_df: pd.DataFrame,
    seq_col: str,
    fit_col: Optional[str],
    ae: torch.nn.Module,
    seq_len: int,
    device: torch.device,
    num_chains: int,
) -> List[torch.Tensor]:
    if fit_col is not None:
        ranked = train_df.dropna(subset=[fit_col]).sort_values(fit_col, ascending=False)
    else:
        ranked = train_df.copy()

    if len(ranked) == 0:
        ranked = train_df.copy()

    ranked = ranked.drop_duplicates(subset=[seq_col], keep="first")
    seeds: List[torch.Tensor] = []
    for _, row in ranked.head(max(num_chains, 8)).iterrows():
        seq = str(row[seq_col])
        z = ae.encode(seq_to_inds(seq, seq_len).unsqueeze(0).to(device)).squeeze(0).to(torch.float32)
        seeds.append(z)
        if len(seeds) >= num_chains:
            break

    if not seeds:
        raise ValueError("Could not build any initialization seeds for ESS/TESS")

    return seeds


@torch.no_grad()
def _ess_step_with_delta(
    z_current: torch.Tensor,
    ll_current: torch.Tensor,
    ll_fn,
    center: Optional[torch.Tensor],
    delta: Optional[float],
    temperature: float,
    max_attempts: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    ESS step with optional ball constraint (delta) around center.
    Wraps sampler_ess.ess_step for use inside sample_ess_tess.
    """
    nu = temperature * torch.randn_like(z_current)
    log_y = ll_current + torch.log(torch.rand(1, device=z_current.device))

    theta = torch.rand(1, device=z_current.device) * (2.0 * math.pi)
    theta_min = theta - 2.0 * math.pi
    theta_max = theta.clone()

    for attempts in range(1, max_attempts + 1):
        prop = z_current * torch.cos(theta) + nu * torch.sin(theta)
        if center is not None and delta is not None:
            if torch.norm(prop - center, p=2).item() > float(delta):
                if theta.item() < 0:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = torch.empty(1, device=z_current.device).uniform_(theta_min.item(), theta_max.item())
                continue
        ll_prop = ll_fn(prop.unsqueeze(0)).squeeze(0)
        if ll_prop > log_y:
            return prop, ll_prop, attempts
        if theta.item() < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = torch.empty(1, device=z_current.device).uniform_(theta_min.item(), theta_max.item())

    return z_current, ll_current, max_attempts


@torch.no_grad()
def sample_ess_tess(
    ae: torch.nn.Module,
    ll_fn,
    z_starts: List[torch.Tensor],
    n: int,
    num_chains: int,
    burnin: int,
    max_steps: int,
    center: Optional[torch.Tensor],
    delta: Optional[float],
    delta_final: Optional[float],
    temperature: float,
    use_transport: bool,
    transport_strength: float,
    latent_dim: int = 32,
    flow_buffer_size: int = 128,
    flow_adapt_every: int = 32,
    flow_lr: float = 1e-3,
    flow_adapt_steps: int = 5,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    rows = []
    duplicate_rows = []
    per_chain = math.ceil(n / num_chains)
    seen_sequences = set()

    t_start = time.perf_counter()

    for chain in range(num_chains):
        z = z_starts[chain % len(z_starts)].clone()
        ll = ll_fn(z.unsqueeze(0)).squeeze(0)
        saved = 0

        # Build a per-chain TransportESSSampler when use_transport is True
        if use_transport:
            tess_sampler = TransportESSSampler(
                z_init=z,
                log_likelihood_fn=ll_fn,
                latent_dim=latent_dim,
                temperature=temperature,
                buffer_size=flow_buffer_size,
                adapt_every=flow_adapt_every,
                flow_lr=flow_lr,
                n_adapt_steps=flow_adapt_steps,
                device=device or z.device,
            )

        for step in range(max_steps):
            if delta is not None and delta_final is not None and max_steps > 1:
                frac = float(step) / float(max_steps - 1)
                delta_now = (1.0 - frac) * float(delta) + frac * float(delta_final)
            else:
                delta_now = delta

            if use_transport:
                z, accepted = tess_sampler.step()
                ll = ll_fn(z.unsqueeze(0)).squeeze(0)
            else:
                z, ll, _ = _ess_step_with_delta(
                    z, ll, ll_fn, center, delta_now, temperature
                )

            if step < burnin:
                continue

            seq = clean_seq(inds_to_seq(torch.argmax(ae.decode(z.unsqueeze(0)), dim=1).squeeze(0)))
            if seq in seen_sequences:
                duplicate_rows.append({"chain": chain, "step": step, "sequence": seq, "score": float(ll.item())})
                continue
            seen_sequences.add(seq)
            rows.append({"chain": chain, "step": step, "sequence": seq, "score": float(ll.item())})
            saved += 1
            if saved >= per_chain:
                break

    runtime_sec = time.perf_counter() - t_start

    if len(rows) < n and duplicate_rows:
        need = n - len(rows)
        duplicate_rows = sorted(duplicate_rows, key=lambda r: r["score"], reverse=True)
        rows.extend(duplicate_rows[:need])

    out = pd.DataFrame(rows).head(n).reset_index(drop=True)
    out["runtime_sec"] = runtime_sec  # written for ESS/sec computation in evaluate_method
    if len(out) < n:
        print(
            f"[warn] sampler returned {len(out)} < requested {n} (insufficient accepted states under current constraints)"
        )
    return out


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {args.alpha}")
    if args.latent_temperature <= 0:
        raise ValueError(f"latent-temperature must be > 0, got {args.latent_temperature}")
    if args.delta_final is not None and args.delta_final <= 0:
        raise ValueError(f"delta-final must be > 0 when provided, got {args.delta_final}")
    if args.transport_strength < 0:
        raise ValueError(f"transport-strength must be >= 0, got {args.transport_strength}")
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    proldm_root = (script_dir / args.proldm_root).resolve()
    train_csv = (proldm_root / args.train_csv).resolve()
    baseline_ckpt = (proldm_root / args.baseline_ckpt).resolve()
    ae_ckpt = (proldm_root / args.ae_ckpt).resolve()
    out_dir = (script_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [proldm_root, train_csv, baseline_ckpt, ae_ckpt]:
        if not p.exists():
            raise FileNotFoundError(p)

    validate_model_type(args.model_type, ae_ckpt, label="AE/JTAE")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    train_df = pd.read_csv(train_csv)
    seq_col = infer_seq_col(train_df)
    wt_seq = choose_wt(train_df)

    sys.path.insert(0, str(proldm_root))
    from model.JTAE.models_condif_1d import jtae

    ae_raw = torch.load(ae_ckpt, map_location=device)
    ae_state = normalize_ckpt_state(ae_raw)
    dims = infer_dims_from_state(ae_state)
    hparams = build_hparams(args.dataset, dims, device, batch_size=16)
    ae = jtae(hparams).to(device)
    ae.load_state_dict(filter_by_shape(ae_state, ae.state_dict()), strict=False)
    ae.eval()

    fit_col = infer_fitness_col(train_df)
    if fit_col is not None:
        ranked = train_df.dropna(subset=[fit_col]).sort_values(fit_col, ascending=False)
        if len(ranked) == 0:
            start_seq = str(train_df.iloc[0][seq_col])
            print(f"[init] fitness column '{fit_col}' had no valid rows; falling back to first training sequence")
        else:
            start_seq = str(ranked.iloc[0][seq_col])
            print(f"[init] using highest-fitness training sequence from column '{fit_col}'")
    else:
        start_seq = str(train_df.iloc[0][seq_col])
        print("[init] no fitness column found; falling back to first training sequence")

    z_start = ae.encode(seq_to_inds(start_seq, dims["seq_len"]).unsqueeze(0).to(device)).squeeze(0).to(torch.float32)
    z_wt = ae.encode(seq_to_inds(wt_seq, dims["seq_len"]).unsqueeze(0).to(device)).squeeze(0).to(torch.float32)
    z_start_pool = build_chain_starts(train_df, seq_col, fit_col, ae, dims["seq_len"], device, args.num_chains)

    esm2 = ESM2Scorer(args.esm2_model, device=device, head_path=args.esm2_head_path) if args.use_esm2 else None
    ll = Likelihood(ae, esm2, dims["seq_len"], device, esm_w=args.esm_weight, reg_w=args.reg_weight, alpha=args.alpha)
    if args.use_esm2:
        print(f"[likelihood] hybrid score = alpha*ESM2 + (1-alpha)*Regressor with alpha={args.alpha:.3f}")
    else:
        print("[likelihood] regressor-only score (ESM2 disabled)")
    if args.use_transport:
        print(f"[sampler] transport ESS with learned RealNVP flow (buffer={args.flow_buffer_size}, adapt_every={args.flow_adapt_every})")
    else:
        print("[sampler] standard ESS")

    latent_dim = dims.get("latent_dim", 32)
    modes = [args.mode] if args.mode != "all" else ["baseline", "ess", "tess", "transport_ess"]

    if "baseline" in modes:
        validate_model_type(args.baseline_model_type, baseline_ckpt, label="Baseline diffusion")
        baseline_seqs = diffusion_baseline(
            proldm_root=proldm_root,
            ckpt_path=baseline_ckpt,
            dataset=args.dataset,
            n=args.n,
            label=8,
            omega=args.omega,
            device=device,
        )
        pd.DataFrame({"sequence": baseline_seqs}).to_csv(out_dir / "baseline_pldm.csv", index=False)

    if "ess" in modes:
        ess_df = sample_ess_tess(
            ae=ae, ll_fn=ll, z_starts=z_start_pool, n=args.n, num_chains=args.num_chains,
            burnin=args.burnin, max_steps=args.max_steps,
            center=z_start, delta=args.tess_delta, delta_final=args.delta_final,
            temperature=args.latent_temperature, use_transport=False,
            transport_strength=args.transport_strength, latent_dim=latent_dim, device=device,
        )
        ess_df.to_csv(out_dir / "results_ess.csv", index=False)

    if "tess" in modes:
        tess_df = sample_ess_tess(
            ae=ae, ll_fn=ll, z_starts=z_start_pool, n=args.n, num_chains=args.num_chains,
            burnin=args.burnin, max_steps=args.max_steps,
            center=z_wt, delta=args.tess_delta, delta_final=args.delta_final,
            temperature=args.latent_temperature, use_transport=False,
            transport_strength=args.transport_strength, latent_dim=latent_dim, device=device,
        )
        tess_df.to_csv(out_dir / "results_tess.csv", index=False)

    if "transport_ess" in modes:
        transport_df = sample_ess_tess(
            ae=ae, ll_fn=ll, z_starts=z_start_pool, n=args.n, num_chains=args.num_chains,
            burnin=args.burnin, max_steps=args.max_steps,
            center=z_wt, delta=args.tess_delta, delta_final=args.delta_final,
            temperature=args.latent_temperature, use_transport=True,
            transport_strength=args.transport_strength, latent_dim=latent_dim,
            flow_buffer_size=args.flow_buffer_size, flow_adapt_every=args.flow_adapt_every,
            flow_lr=args.flow_lr, flow_adapt_steps=args.flow_adapt_steps, device=device,
        )
        transport_df.to_csv(out_dir / "results_transport_ess.csv", index=False)

    print(f"Saved raw sets to: {out_dir}")


if __name__ == "__main__":
    main()
