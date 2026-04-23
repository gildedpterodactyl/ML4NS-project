import argparse
import math
import random
import sys
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
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--num-chains", type=int, default=8)
    p.add_argument("--tess-delta", type=float, default=6.0)
    p.add_argument("--burnin", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--mode", type=str, choices=["baseline", "ess", "tess", "all"], default="all")
    p.add_argument("--omega", type=float, default=20.0)
    p.add_argument("--esm-weight", type=float, default=0.6)
    p.add_argument("--reg-weight", type=float, default=0.4)
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
    def __init__(self, ae: torch.nn.Module, esm2: ESM2Scorer, seq_len: int, device: torch.device, esm_w: float, reg_w: float):
        self.ae = ae
        self.esm2 = esm2
        self.seq_len = seq_len
        self.device = device
        self.esm_w = esm_w
        self.reg_w = reg_w

    @torch.no_grad()
    def decode_seq(self, z: torch.Tensor) -> List[str]:
        logits = self.ae.decode(z)
        idx = torch.argmax(logits, dim=1)
        return [clean_seq(inds_to_seq(x)) for x in idx]

    @torch.no_grad()
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        reg = self.ae.regressor_module(z).squeeze(-1)
        seqs = self.decode_seq(z)
        esm, _ = self.esm2.score_and_perplexity(seqs)
        reg_z = (reg - reg.mean()) / reg.std(unbiased=False).clamp_min(1e-8)
        esm_z = (esm - esm.mean()) / esm.std(unbiased=False).clamp_min(1e-8)
        return self.esm_w * esm_z + self.reg_w * reg_z


@torch.no_grad()
def ess_step(
    z_current: torch.Tensor,
    ll_current: torch.Tensor,
    ll_fn,
    center: Optional[torch.Tensor],
    delta: Optional[float],
    max_attempts: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    nu = torch.randn_like(z_current)
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
    z_start: torch.Tensor,
    n: int,
    num_chains: int,
    burnin: int,
    max_steps: int,
    center: Optional[torch.Tensor],
    delta: Optional[float],
) -> pd.DataFrame:
    rows = []
    per_chain = math.ceil(n / num_chains)

    for chain in range(num_chains):
        z = z_start.clone()
        ll = ll_fn(z.unsqueeze(0)).squeeze(0)
        saved = 0
        for step in range(max_steps):
            z, ll, attempts = ess_step(z, ll, ll_fn, center, delta)
            if step < burnin:
                continue
            seq = clean_seq(inds_to_seq(torch.argmax(ae.decode(z.unsqueeze(0)), dim=1).squeeze(0)))
            rows.append({"chain": chain, "step": step, "attempts": attempts, "sequence": seq, "score": float(ll.item())})
            saved += 1
            if saved >= per_chain:
                break

    out = pd.DataFrame(rows).head(n).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
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

    start_seq = str(train_df.iloc[0][seq_col])
    z_start = ae.encode(seq_to_inds(start_seq, dims["seq_len"]).unsqueeze(0).to(device)).squeeze(0).to(torch.float32)
    z_wt = ae.encode(seq_to_inds(wt_seq, dims["seq_len"]).unsqueeze(0).to(device)).squeeze(0).to(torch.float32)

    esm2 = ESM2Scorer(args.esm2_model, device=device, head_path=args.esm2_head_path)
    ll = Likelihood(ae, esm2, dims["seq_len"], device, esm_w=args.esm_weight, reg_w=args.reg_weight)

    modes = [args.mode] if args.mode != "all" else ["baseline", "ess", "tess"]
    if "baseline" in modes:
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
            center=None,
            delta=None,
        )
        ess_df.to_csv(out_dir / "results_ess.csv", index=False)

    if "tess" in modes:
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
        )
        tess_df.to_csv(out_dir / "results_tess.csv", index=False)

    print(f"Saved raw sets to: {out_dir}")


if __name__ == "__main__":
    main()
