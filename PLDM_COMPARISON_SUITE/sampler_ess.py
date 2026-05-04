"""
Sampling utilities:
  - Elliptical Slice Sampling (ESS)
  - Simplified Transport ESS (TESS)

ESS follows Murray et al. (2010) exactly.

TESS uses a *learnable* normalising flow (RealNVP-style affine coupling
layers) as a preconditioning transport map, approximating Algorithm 1 of
Cabezas & Zanella (2023).  The flow is adapted online using the last
`flow_buffer_size` accepted samples.

Changes vs previous version
----------------------------
- Temperature now scales the *initial* ellipse auxiliary draw only.
  The bracket-shrinkage loop still uses the true (temperature-corrected)
  log-likelihood threshold so the chain targets the tempered posterior
  consistently.  T=1 recovers the exact ESS stationary distribution.
- Likelihood.forward now uses *running* mean/std over an EMA buffer
  instead of single-sample normalisation (avoids std=0 at batch-size 1).
- TESS: replaced the fixed-elementwise preconditioning with a learned
  2-layer RealNVP flow updated via KL-divergence on recent accepted samples.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
#  Likelihood wrapper
# ============================================================

class RunningNorm:
    """Online EMA mean/std for scalar streams (no batch-size-1 std=0 issue)."""

    def __init__(self, alpha: float = 0.05, eps: float = 1e-6) -> None:
        self.alpha = alpha
        self.eps = eps
        self._mean: Optional[torch.Tensor] = None
        self._var:  Optional[torch.Tensor] = None

    def update_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Update running stats with *x* and return normalised *x*."""
        if self._mean is None:
            self._mean = x.detach()
            self._var  = torch.ones_like(x.detach())
            return torch.zeros_like(x)
        self._mean = (1 - self.alpha) * self._mean + self.alpha * x.detach()
        self._var  = (1 - self.alpha) * self._var  + self.alpha * (x.detach() - self._mean) ** 2
        return (x - self._mean) / (self._var.sqrt() + self.eps)


class Likelihood(nn.Module):
    """
    Combined fitness likelihood used as the log-likelihood for ESS/TESS.

    score = (1-alpha) * norm(regressor(z)) + alpha * norm(-perplexity(z))

    Running EMA normalisation is used so that single-sample evaluation
    (batch size = 1) is numerically stable.
    """

    def __init__(
        self,
        regressor: nn.Module,
        esm2_scorer: Optional[nn.Module] = None,
        alpha: float = 0.0,
        ema_alpha: float = 0.05,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.regressor   = regressor
        self.esm2_scorer = esm2_scorer
        self.alpha       = alpha
        self.temperature = temperature

        self._norm_reg  = RunningNorm(alpha=ema_alpha)
        self._norm_esm  = RunningNorm(alpha=ema_alpha)

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return scalar log-likelihood for a *single* latent vector z (1-D)."""
        z_ = z.unsqueeze(0)                         # (1, D)
        reg_score = self.regressor(z_).squeeze()    # scalar

        if self.esm2_scorer is not None and self.alpha > 0:
            esm_score = self.esm2_scorer(z_).squeeze()
            reg_n  = self._norm_reg.update_and_normalize(reg_score)
            esm_n  = self._norm_esm.update_and_normalize(esm_score)
            score  = (1.0 - self.alpha) * reg_n + self.alpha * esm_n
        else:
            score = self._norm_reg.update_and_normalize(reg_score)

        return score / self.temperature


# ============================================================
#  Minimal RealNVP flow (2-layer affine coupling)
# ============================================================

class _AffineCoupling(nn.Module):
    def __init__(self, d: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        d_in = int(mask.sum().item())
        d_out = d - d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, 2 * d),
            nn.Tanh(),
            nn.Linear(2 * d, 2 * d_out),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        x_in = x_masked[:, self.mask.bool()]
        h = self.net(x_in)
        d_out = h.size(1) // 2
        s_raw, t = h[:, :d_out], h[:, d_out:]
        s = torch.tanh(s_raw)
        y = x.clone()
        y[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_masked = y * self.mask
        y_in = y_masked[:, self.mask.bool()]
        h = self.net(y_in)
        d_out = h.size(1) // 2
        s_raw, t = h[:, :d_out], h[:, d_out:]
        s = torch.tanh(s_raw)
        x = y.clone()
        x[:, ~self.mask.bool()] = (y[:, ~self.mask.bool()] - t) * torch.exp(-s)
        return x


class RealNVPFlow(nn.Module):
    """
    Lightweight 2-layer RealNVP as the transport map T for TESS.

    T maps the latent z ~ pi(.) to a whitened space u ~ N(0,I).
    Adapted online (self.adapt) using recent accepted samples.
    """

    def __init__(self, d: int, lr: float = 1e-3, n_adapt_steps: int = 5) -> None:
        super().__init__()
        m1 = torch.zeros(d, dtype=torch.bool)
        m1[:d // 2] = True
        m2 = ~m1
        self.c1 = _AffineCoupling(d, m1)
        self.c2 = _AffineCoupling(d, m2)
        self._opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.n_adapt_steps = n_adapt_steps

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """z -> u, log|det J|"""
        u, ld1 = self.c1(z)
        u, ld2 = self.c2(u)
        return u, ld1 + ld2

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """u -> z"""
        z = self.c2.inverse(u)
        z = self.c1.inverse(z)
        return z

    def adapt(self, samples: torch.Tensor) -> float:
        """One step of maximum-likelihood (NLL) adaptation on *samples* (N,D)."""
        total_loss = 0.0
        self.train()
        for _ in range(self.n_adapt_steps):
            self._opt.zero_grad()
            u, log_det = self.forward(samples)
            nll = (0.5 * (u ** 2).sum(dim=-1) - log_det).mean()
            nll.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self._opt.step()
            total_loss += nll.item()
        self.eval()
        return total_loss / self.n_adapt_steps


# ============================================================
#  ESS
# ============================================================

def ess_step(
    z: torch.Tensor,
    log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
    temperature: float = 1.0,
    max_iters: int = 100,
) -> Tuple[torch.Tensor, bool]:
    """
    Single Elliptical Slice Sampling step (Murray et al. 2010).

    Parameters
    ----------
    z               : current latent (D,)
    log_likelihood_fn : log L(z) -- may include temperature scaling
    temperature     : scale auxiliary nu ~ N(0, T^2 I); T=1 targets true posterior
    max_iters       : bracket shrinkage iteration limit

    Returns
    -------
    z_new, accepted  (accepted=False only on max_iters reached)
    """
    nu  = temperature * torch.randn_like(z)

    with torch.no_grad():
        log_y = log_likelihood_fn(z) + torch.log(torch.rand(1, device=z.device))

    theta     = torch.rand(1).item() * 2 * math.pi
    theta_min = theta - 2 * math.pi
    theta_max = theta

    for _ in range(max_iters):
        z_prop = z * math.cos(theta) + nu * math.sin(theta)
        with torch.no_grad():
            if log_likelihood_fn(z_prop) > log_y:
                return z_prop, True
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = torch.rand(1).item() * (theta_max - theta_min) + theta_min

    warnings.warn("ESS bracket shrinkage did not converge; returning current point.")
    return z, False


# ============================================================
#  Transport ESS (learned flow)
# ============================================================

class TransportESSSampler:
    """
    Transport ESS with an online-adapted RealNVP preconditioning flow.

    The flow is updated every `adapt_every` accepted samples using the
    last `buffer_size` accepted latents, approximating Algorithm 2 of
    Cabezas & Zanella (2023).
    """

    def __init__(
        self,
        z_init: torch.Tensor,
        log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
        latent_dim: int,
        temperature: float = 1.0,
        buffer_size: int = 128,
        adapt_every: int = 32,
        flow_lr: float = 1e-3,
        n_adapt_steps: int = 5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.z   = z_init.clone()
        self.ll  = log_likelihood_fn
        self.T   = temperature
        self.buf: deque[torch.Tensor] = deque(maxlen=buffer_size)
        self.buf.append(z_init.detach().cpu())
        self.adapt_every = adapt_every
        self._since_adapt = 0
        self._device = device or z_init.device

        self.flow = RealNVPFlow(latent_dim, lr=flow_lr, n_adapt_steps=n_adapt_steps).to(self._device)

    def step(self) -> Tuple[torch.Tensor, bool]:
        """One TESS step: map to whitened space, run ESS, map back."""
        with torch.no_grad():
            u, _ = self.flow(self.z.unsqueeze(0))
        u = u.squeeze(0)

        def ll_u(u_: torch.Tensor) -> torch.Tensor:
            u_b = u_.unsqueeze(0)
            z_  = self.flow.inverse(u_b).squeeze(0)
            _, log_det = self.flow(u_b)
            return self.ll(z_) + log_det.squeeze()

        u_new, accepted = ess_step(u, ll_u, temperature=self.T)

        with torch.no_grad():
            z_new = self.flow.inverse(u_new.unsqueeze(0)).squeeze(0)

        if accepted:
            self.z = z_new
            self.buf.append(z_new.detach().cpu())
            self._since_adapt += 1
            if self._since_adapt >= self.adapt_every and len(self.buf) >= 16:
                batch = torch.stack(list(self.buf)).to(self._device)
                self.flow.adapt(batch)
                self._since_adapt = 0

        return self.z, accepted
