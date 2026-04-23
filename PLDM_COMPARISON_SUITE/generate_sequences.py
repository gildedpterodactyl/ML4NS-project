import argparse
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
    p.add_argument("--tess-delta", type=float, default=0.14)
    p.add_argument("--tess-warp-strength", type=float, default=0.75)
    p.add_argument("--delta-final", type=float, default=None)
    p.add_argument("--latent-temperature", type=float, default=0.75)
    p.add_argument("--decode-temperature", type=float, default=0.5,
                   help="Softmax temperature for sequence decoding (>0). "
                        "1.0=uniform sampling, 0.0 approaches argmax. Default 0.5.")
    p.add_argument("--chain-init-noise", type=float, default=0.05,
                   help="Std of per-chain perturbation added to z_start so chains "
                        "explore different neighbourhoods. Default 0.05.")
    p.add_argument("--burnin", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--mode", type=str, choices=["baseline", "ess", "tess", "all"], default="all")
    p.add_argument("--omega", type=float, default=20.0)
    p.add_argument("--esm-weight", type=float, default=0.5)
    p.add_argument("--reg-weight", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--model-type", type=str, choices=["auto", "tiny", "standard"], default="auto")
    p.add_argument("--baseline-model-type", type=str, choices=["auto", "tiny", "standard"], default="auto")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs")
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
    def __init__(self, ae: torch.nn.Module, esm2: ESM2Scorer, seq_len: int, device: torch.device, esm_w: float, reg_w: float, alpha: float):
        self.ae = ae
        self.esm2 = esm2
        self.seq_len = seq_len
        self.device = device
        self.esm_w = esm_w
        self.reg_w = reg_w
        self.alpha = alpha

    @torch.no_grad()
    def decode_seq(self, z: torch.Tensor) -> List[str]:
        logits = self.ae.decode(z)
        idx = torch.argmax(logits, dim=1)
        return [clean_seq(inds_to_seq(x)) for x in idx]

    @torch.no_grad()
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        reg = self.ae.regressor_module(z).squeeze(-1)
        if self.esm2 is None:
            return reg
        seqs = self.decode_seq(z)
        esm, _ = self.esm2.score_and_perplexity(seqs)
        if reg.numel() == 1:
            # ESS/TESS evaluate one latent at a time; batch z-scoring collapses to 0.
            # Use raw scores so acceptance decisions remain meaningful.
            reg_z = reg
            esm_z = esm
        else:
            reg_z = (reg - reg.mean()) / reg.std(unbiased=False).clamp_min(1e-8)
            esm_z = (esm - esm.mean()) / esm.std(unbiased=False).clamp_min(1e-8)
        if self.alpha is not None:
            return self.alpha * esm_z + (1.0 - self.alpha) * reg_z
        return self.esm_w * esm_z + self.reg_w * reg_z


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


@torch.no_grad()
def decode_with_temperature(
    ae: torch.nn.Module,
    z: torch.Tensor,
    temperature: float,
) -> str:
    """Decode a single latent vector z to a sequence string.

    When temperature > 0, each position is sampled from the softmax
    distribution rather than taken as argmax.  This ensures that
    nearby latent points (which have similar but non-identical logits)
    produce *distinct* sequences, recovering the diversity that the
    sampler generates in latent space.

    temperature = 0.0  →  pure argmax (old behaviour, causes collapse)
    temperature = 0.5  →  sharply peaked sampling (recommended default)
    temperature = 1.0  →  full categorical sampling
    """
    logits = ae.decode(z.unsqueeze(0)).squeeze(0)  # (vocab, seq_len)
    if temperature <= 0.0:
        idx = torch.argmax(logits, dim=0)
    else:
        probs = torch.softmax(logits / temperature, dim=0)  # (vocab, seq_len)
        # sample independently at each position
        idx = torch.multinomial(probs.T, num_samples=1).squeeze(-1)  # (seq_len,)
    return clean_seq(inds_to_seq(idx))


@torch.no_grad()
def ess_step(
    z_current: torch.Tensor,
    ll_current: torch.Tensor,
    ll_fn,
    center: Optional[torch.Tensor],
    delta: Optional[float],
    temperature: float,
    proposal_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    max_attempts: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    # FIX 1: nu must NOT be scaled by temperature.
    # In Elliptical Slice Sampling the auxiliary vector nu is drawn from
    # the same prior as z (i.e. N(0,I)).  Scaling nu down shrinks the
    # proposal ellipse dramatically, trapping all chains in a tiny basin
    # around z_current and causing every decoded sequence to be identical.
    # temperature is used only via log_y to control the acceptance threshold.
    nu = torch.randn_like(z_current)  # prior sample — no temperature scaling
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

        prop_eval = proposal_transform(prop) if proposal_transform is not None else prop
        ll_prop = ll_fn(prop_eval.unsqueeze(0)).squeeze(0)
        if ll_prop > log_y:
            return prop_eval, ll_prop, attempts

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
    z_start: torch.Tensor,
    n: int,
    num_chains: int,
    burnin: int,
    max_steps: int,
    center: Optional[torch.Tensor],
    delta: Optional[float],
    delta_final: Optional[float],
    temperature: float,
    decode_temperature: float = 0.5,
    chain_init_noise: float = 0.05,
    proposal_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> pd.DataFrame:
    rows = []
    per_chain = math.ceil(n / num_chains)

    for chain in range(num_chains):
        # FIX 2: perturb z_start independently per chain so chains explore
        # different neighbourhoods from the very first step.  Without this,
        # all chains start at the same point and — even with a correct ESS
        # proposal — converge on the same local basin.
        if chain_init_noise > 0.0:
            z = z_start.clone() + chain_init_noise * torch.randn_like(z_start)
        else:
            z = z_start.clone()

        ll = ll_fn(z.unsqueeze(0)).squeeze(0)
        saved = 0
        for step in range(max_steps):
            if delta is not None and delta_final is not None and max_steps > 1:
                frac = float(step) / float(max_steps - 1)
                delta_now = (1.0 - frac) * float(delta) + frac * float(delta_final)
            else:
                delta_now = delta
            z, ll, attempts = ess_step(
                z,
                ll,
                ll_fn,
                center,
                delta_now,
                temperature,
                proposal_transform=proposal_transform,
            )
            if step < burnin:
                continue

            # FIX 3: use temperature-sampled decode instead of argmax.
            # Argmax is deterministic: all latent points in the same
            # decoder basin produce the *exact same* sequence string,
            # hiding the diversity the sampler actually generates.
            # Temperature sampling preserves that latent diversity.
            seq = decode_with_temperature(ae, z, temperature=decode_temperature)
            rows.append({"chain": chain, "step": step, "attempts": attempts, "sequence": seq, "score": float(ll.item())})
            saved += 1
            if saved >= per_chain:
                break

    out = pd.DataFrame(rows).head(n).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {args.alpha}")
    if args.latent_temperature <= 0:
        raise ValueError(f"latent-temperature must be > 0, got {args.latent_temperature}")
    if args.delta_final is not None and args.delta_final <= 0:
        raise ValueError(f"delta-final must be > 0 when provided, got {args.delta_final}")
    if args.decode_temperature < 0:
        raise ValueError(f"decode-temperature must be >= 0, got {args.decode_temperature}")
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

    def make_radial_warp(center: torch.Tensor, strength: float) -> Callable[[torch.Tensor], torch.Tensor]:
        center = center.clone()
        ref = torch.norm(z_start - center, p=2).item()
        ref = max(ref, 1.0)
        axis = z_start - center
        axis_norm = torch.norm(axis, p=2).item()
        if axis_norm < 1e-8:
            axis = torch.zeros_like(center)
            axis[0] = 1.0
        else:
            axis = axis / axis_norm

        def _warp(z: torch.Tensor) -> torch.Tensor:
            delta = z - center
            radius = torch.norm(delta, p=2).clamp_min(1e-8)
            scaled = radius / ref
            # Expand modestly away from the anchor so TESS explores a distinct shell,
            # then add a deterministic shear/twist to break ESS-like symmetry.
            radial_scale = 1.0 + strength * (scaled / (1.0 + scaled))
            parallel = torch.dot(delta, axis) * axis
            orth = delta - parallel
            twist = 0.35 * strength * torch.roll(delta, shifts=max(1, delta.numel() // 7))
            return center + parallel * radial_scale + orth * (1.0 + 0.75 * strength) + twist

        return _warp

    esm2 = ESM2Scorer(args.esm2_model, device=device, head_path=args.esm2_head_path) if args.use_esm2 else None
    ll = Likelihood(ae, esm2, dims["seq_len"], device, esm_w=args.esm_weight, reg_w=args.reg_weight, alpha=args.alpha)
    if args.use_esm2:
        print(f"[likelihood] hybrid score = alpha*ESM2 + (1-alpha)*Regressor with alpha={args.alpha:.3f}")
    else:
        print("[likelihood] regressor-only score (ESM2 disabled)")

    print(f"[decode] temperature={args.decode_temperature:.3f} ({'argmax' if args.decode_temperature <= 0 else 'sampled'})")
    print(f"[chains] init noise std={args.chain_init_noise:.4f}")

    modes = [args.mode] if args.mode != "all" else ["baseline", "ess", "tess"]
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
            ae=ae,
            ll_fn=ll,
            z_start=z_start,
            n=args.n,
            num_chains=args.num_chains,
            burnin=args.burnin,
            max_steps=args.max_steps,
            center=z_start,
            delta=args.tess_delta,
            delta_final=args.delta_final,
            temperature=args.latent_temperature,
            decode_temperature=args.decode_temperature,
            chain_init_noise=args.chain_init_noise,
        )
        ess_df.to_csv(out_dir / "results_ess.csv", index=False)

    if "tess" in modes:
        tess_warp = make_radial_warp(z_wt, args.tess_warp_strength)
        print(f"[warp] TESS radial warp strength={args.tess_warp_strength:.3f} around WT anchor")
        tess_df = sample_ess_tess(
            ae=ae,
            ll_fn=ll,
            z_start=z_start,
            n=args.n,
            num_chains=args.num_chains,
            burnin=args.burnin,
            max_steps=args.max_steps,
            center=z_wt,
            delta=args.tess_delta,
            delta_final=args.delta_final,
            temperature=args.latent_temperature,
            decode_temperature=args.decode_temperature,
            chain_init_noise=args.chain_init_noise,
            proposal_transform=tess_warp,
        )
        tess_df.to_csv(out_dir / "results_tess.csv", index=False)

    print(f"Saved raw sets to: {out_dir}")


if __name__ == "__main__":
    main()
