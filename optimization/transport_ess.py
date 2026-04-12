#!/usr/bin/env python3
"""
Transport Elliptical Slice Sampling (TESS)

Implementation of Algorithm 1 and Algorithm 2 from:
  Cabezas & Nemeth (2023). Transport Elliptical Slice Sampling.
  AISTATS 2023. arXiv:2210.10644v2

Key idea:
  Instead of running ESS directly in latent space z (which may be non-Gaussian),
  TESS learns a normalizing flow T that maps z → u ≈ N(0, I), then runs ESS
  in the transformed (approximately Gaussian) reference space u, and finally
  maps accepted samples back to z via x = T(u).

Two-step procedure:
  1. MAP OPTIMIZATION: train a normalizing flow T (affine coupling layers)
     that minimises KL(T^{-1}_# π || N(0,I)) using warm-up ESS samples.
  2. SAMPLING: run standard ESS in the reference space u, accepting proposals
     u' = u·cos(θ) + v·sin(θ) according to the slice criterion on
     log π(T(u')) + log |det J_T(u')|, then return x = T(u).

Warm-start note
---------------
Same issue as ESS: when target Rg << oracle intercept, the starting u
has terrible score and the warm-up chain freezes.  We run a short
gradient-ascent step in z-space before the warm-up to initialise u in a
reasonable region.  This is controlled by warm_start (default True).

References:
  - Cabezas & Nemeth, AISTATS 2023, arXiv:2210.10644v2
  - Murray, Adams, MacKay. Elliptical Slice Sampling. ICML 2010.
  - Dinh, Krueger, Bengio. NICE. arXiv:1410.8516, 2014.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Normalizing Flow: Affine Coupling Layer (NICE / RealNVP style)
# ---------------------------------------------------------------------------

class AffineCouplingLayer(nn.Module):
    """
    One affine coupling block (Dinh et al. 2014, Eq. 7-8 of paper).

    Splits d-dim input into (x_A, x_B) of sizes (d-p, p).
    The coupling function t: R^p → R^(d-p) × R^(d-p) is a 2-hidden-layer MLP.

        y_A = x_A * exp(s(x_B)) + t(x_B)   [scale-shift of first half]
        y_B = x_B                            [identity on second half]

    Forward: u → x  (reference → target space)
    Inverse:  x → u  (target → reference space, needed for KL training)

    Log |det J| = sum(s(x_B))  (sum of log-scales, no abs needed since exp > 0)
    """

    def __init__(self, d: int, d_hidden: int = 32, reverse: bool = False):
        super().__init__()
        self.d = d
        self.reverse = reverse
        self.p = d // 2           # pass-through dimension
        self.q = d - self.p       # transformed dimension

        self.net = nn.Sequential(
            nn.Linear(self.p, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, self.q * 2),
        )
        # Zero-init last layer so T starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.reverse:
            return x[..., :self.q], x[..., self.q:]
        else:
            return x[..., self.p:], x[..., :self.p]

    def _join(self, x_A: torch.Tensor, x_B: torch.Tensor) -> torch.Tensor:
        if not self.reverse:
            return torch.cat([x_A, x_B], dim=-1)
        else:
            return torch.cat([x_B, x_A], dim=-1)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """u → x  (forward / push-forward).  Returns (x, log_det_J)."""
        u_A, u_B = self._split(u)
        st = self.net(u_B)
        s, t_shift = st[..., :self.q], st[..., self.q:]
        # Clamp log-scales to ±5 for numerical stability in early training
        s = torch.clamp(s, -5.0, 5.0)
        x_A = u_A * torch.exp(s) + t_shift
        x = self._join(x_A, u_B)
        log_det = s.sum(dim=-1)
        return x, log_det

    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x → u  (inverse / pull-back).  Returns (u, log_det_J_inv)."""
        x_A, x_B = self._split(x)
        st = self.net(x_B)
        s, t_shift = st[..., :self.q], st[..., self.q:]
        s = torch.clamp(s, -5.0, 5.0)
        u_A = (x_A - t_shift) * torch.exp(-s)
        u = self._join(u_A, x_B)
        log_det_inv = -s.sum(dim=-1)
        return u, log_det_inv


class NormalizingFlow(nn.Module):
    """
    Sequential composition of n_transforms pairs of alternating coupling layers
    (G, D), i.e.  T = D_n ∘ G_n ∘ ... ∘ D_1 ∘ G_1  (paper Sec. 2.5).
    """

    def __init__(self, d: int, n_transforms: int = 2, d_hidden: int = 32):
        super().__init__()
        layers: List[AffineCouplingLayer] = []
        for i in range(n_transforms):
            layers.append(AffineCouplingLayer(d, d_hidden, reverse=False))  # G
            layers.append(AffineCouplingLayer(d, d_hidden, reverse=True))   # D
        self.layers = nn.ModuleList(layers)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """u → x.  Returns (x, total_log_det_J)."""
        x = u
        log_det = torch.zeros(u.shape[0] if u.dim() > 1 else 1, device=u.device)
        for layer in self.layers:
            x, ld = layer(x)
            log_det = log_det + ld
        return x, log_det

    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x → u.  Returns (u, total_log_det_J_inv)."""
        u = x
        log_det_inv = torch.zeros(x.shape[0] if x.dim() > 1 else 1, device=x.device)
        for layer in reversed(self.layers):
            u, ld = layer.inverse(u)
            log_det_inv = log_det_inv + ld
        return u, log_det_inv


# ---------------------------------------------------------------------------
# TESS core: Algorithm 1 (single step) and Algorithm 2 (adaptive)
# ---------------------------------------------------------------------------

def _tess_log_accept(u: torch.Tensor, flow: NormalizingFlow, target_fn) -> float:
    """
    Compute log acceptance value for TESS (Algorithm 1, step 8).

        log π(T(u)) + log |det J_T(u)|

    Clamps the log-det to ±50 to guard against NaN in early training.
    """
    with torch.no_grad():
        u_batch = u.unsqueeze(0)          # [1, d]
        x_batch, log_det = flow(u_batch)  # x = T(u), log|det J_T|
        x = x_batch.squeeze(0)           # [d]
        oracle = target_fn(x)
        log_like = oracle.log_likelihood.item()
        log_det_val = float(torch.clamp(log_det, -50.0, 50.0).item())
    return log_like + log_det_val


def _one_tess_step(
    u: torch.Tensor,
    flow: NormalizingFlow,
    target_fn,
    max_attempts: int = 100,
) -> Tuple[torch.Tensor, float, int]:
    """
    One iteration of Algorithm 1 (Transport ESS step).
    """
    current_log_val = _tess_log_accept(u, flow, target_fn)

    w = random.uniform(0, 1)
    log_s = current_log_val + math.log(w + 1e-300)

    v = torch.randn_like(u)

    theta = random.uniform(0, 2 * math.pi)
    theta_min = theta - 2 * math.pi
    theta_max = theta

    for attempt in range(1, max_attempts + 1):
        u_prime = u * math.cos(theta) + v * math.sin(theta)
        log_val_prime = _tess_log_accept(u_prime, flow, target_fn)

        if log_val_prime > log_s:
            return u_prime, log_val_prime, attempt
        else:
            if theta > 0:
                theta_max = theta
            else:
                theta_min = theta
            theta = random.uniform(theta_min, theta_max)

    return u, current_log_val, max_attempts


def _train_flow(
    flow: NormalizingFlow,
    samples_x: torch.Tensor,
    target_fn,
    m_steps: int = 10,
    lr: float = 1e-3,
) -> None:
    """
    Update flow parameters by minimising the forward KL (Eq. 5 of paper):
        L(φ) = -1/k Σ_i [log p_u(T^{-1}(x_i)) + log |det J_{T^{-1}}(x_i)|]

    Log-determinants are clamped to ±50 for numerical stability.
    """
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    flow.train()

    for _ in range(m_steps):
        optimizer.zero_grad()
        u_batch, log_det_inv = flow.inverse(samples_x)
        # Clamp log-det to prevent NaN in early epochs
        log_det_inv = torch.clamp(log_det_inv, -50.0, 50.0)
        d = u_batch.shape[-1]
        log_p_u = -0.5 * (u_batch ** 2).sum(-1) - 0.5 * d * math.log(2 * math.pi)
        loss = -(log_p_u + log_det_inv).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
        optimizer.step()

    flow.eval()


def run_tess(
    z_start: torch.Tensor,
    target_fn,
    steps: int = 50,
    n_transforms: int = 2,
    d_hidden: int = 32,
    warmup_epochs: int = 3,
    warmup_chains: int = 10,
    m_grad_steps: int = 10,
    flow_lr: float = 1e-3,
    warm_start: bool = True,
    warm_steps: int = 20,
    warm_lr: float = 0.1,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Transport Elliptical Slice Sampling (Algorithm 2 from Cabezas & Nemeth 2023).

    Unlike plain ESS which samples directly in z-space (which may be non-Gaussian),
    TESS first learns a normalizing flow T: u → z that approximately maps the
    prior N(0,I) onto the target distribution π(z), then runs ESS in the
    approximately-Gaussian reference space u.

    Warm-start option (default True): runs warm_steps of gradient ascent
    in z-space to move z_start into a moderate-score region before the
    warm-up ESS chains begin.  This prevents the warm-up from freezing
    when the target Rg is far from the oracle intercept.

    Args:
        z_start:        Initial latent vector [latent_dim]
        target_fn:      Oracle returning OracleOutput.log_likelihood
        steps:          MCMC sampling steps in Phase 2 (default 50)
        n_transforms:   G-D coupling layer pairs in the NF (default 2)
        d_hidden:       Hidden width of coupling MLPs (default 32)
        warmup_epochs:  Warm-up epochs h (default 3)
        warmup_chains:  Chains k per warm-up epoch (default 10)
        m_grad_steps:   Adam steps m per NF update (default 10)
        flow_lr:        Learning rate for Adam NF training (default 1e-3)
        warm_start:     Gradient-ascent pre-conditioning (default True)
        warm_steps:     Pre-conditioning steps (default 20)
        warm_lr:        Pre-conditioning learning rate (default 0.1)

    Returns:
        z_final:    Final sample in latent space
        trajectory: List of accepted log_likelihood values (one per step)
    """
    d = z_start.shape[0]
    device = z_start.device

    # Build and initialise normalising flow (identity at init)
    flow = NormalizingFlow(d=d, n_transforms=n_transforms, d_hidden=d_hidden).to(device)
    flow.eval()

    # Optional gradient-ascent warm-start in z-space
    if warm_start:
        z_ws = z_start.clone().detach().requires_grad_(True)
        opt_ws = torch.optim.Adam([z_ws], lr=warm_lr)
        for _ in range(warm_steps):
            opt_ws.zero_grad()
            (-target_fn(z_ws).log_likelihood).backward()
            opt_ws.step()
        z_init = z_ws.detach()
    else:
        z_init = z_start.clone().detach()

    # Initialise u = T^{-1}(z_init) — identity at start → u ≈ z_init
    with torch.no_grad():
        u_batch, _ = flow.inverse(z_init.unsqueeze(0))
    u = u_batch.squeeze(0).detach()

    # -------------------------------------------------------------------
    # PHASE 1: Warm-up — alternate sampling and flow training
    # -------------------------------------------------------------------
    print(f"  [TESS] Warm-up: {warmup_epochs} epochs × {warmup_chains} chains")
    for epoch in range(warmup_epochs):
        warmup_x_list = []
        u_chain = u.clone()
        for _ in range(warmup_chains):
            u_chain, _, _ = _one_tess_step(u_chain, flow, target_fn)
            with torch.no_grad():
                x_batch, _ = flow(u_chain.unsqueeze(0))
            warmup_x_list.append(x_batch.squeeze(0).detach())
        warmup_x = torch.stack(warmup_x_list, dim=0)   # [k, d]

        _train_flow(flow, warmup_x, target_fn, m_steps=m_grad_steps, lr=flow_lr)

        # Re-encode current z under updated flow
        with torch.no_grad():
            x_cur_batch, _ = flow(u.unsqueeze(0))
            u_batch_new, _ = flow.inverse(x_cur_batch)
        u = u_batch_new.squeeze(0).detach()

        print(f"  [TESS]  Warm-up epoch {epoch + 1}/{warmup_epochs} done")

    # -------------------------------------------------------------------
    # PHASE 2: Sampling with fixed T (Algorithm 1 for `steps` iterations)
    # -------------------------------------------------------------------
    trajectory: List[float] = []
    print(f"  [TESS] Sampling phase: {steps} steps")

    for step in range(steps):
        u, _, attempts = _one_tess_step(u, flow, target_fn)

        with torch.no_grad():
            x_batch, _ = flow(u.unsqueeze(0))
            x = x_batch.squeeze(0)
            oracle_out = target_fn(x)
            log_score = oracle_out.log_likelihood.item()

        trajectory.append(log_score)

        if (step + 1) % max(1, steps // 10) == 0 or step == 0:
            print(
                f"  [TESS]  Step {step + 1:3d}/{steps} | "
                f"log_score: {log_score:8.4f} | "
                f"||u||: {torch.norm(u).item():7.4f} | "
                f"attempts: {attempts}"
            )

    # Return final z = T(u_final)
    with torch.no_grad():
        z_final_batch, _ = flow(u.unsqueeze(0))
    z_final = z_final_batch.squeeze(0).detach()

    return z_final, trajectory


if __name__ == "__main__":
    print("Testing run_tess...")
    from optimization.target_function import TargetFunction
    from pathlib import Path

    checkpoint_path = Path("regression/outputs/ridge_latent_tm_model.npz")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        exit(1)

    target_fn = TargetFunction.from_checkpoint(
        model_path=checkpoint_path,
        target_value=65.0,
    )
    latent_dim = target_fn.get_latent_dim()
    z_start = torch.randn(latent_dim)

    print(f"\nRunning TESS (latent_dim={latent_dim})...")
    z_final, trajectory = run_tess(
        z_start=z_start,
        target_fn=target_fn,
        steps=50,
    )

    print(f"\nResults:")
    print(f"  Initial log_score: {trajectory[0]:.4f}")
    print(f"  Final log_score:   {trajectory[-1]:.4f}")
    print(f"  Mean (last 10):    {sum(trajectory[-10:]) / 10:.4f}")
    print(f"  Max score:         {max(trajectory):.4f}")
    print(f"  ||z_final||:       {torch.norm(z_final).item():.4f}")
