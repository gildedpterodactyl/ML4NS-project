#!/usr/bin/env python3
"""
Algorithm 2: Elliptical Slice Sampling (ESS)

ESS is an MCMC algorithm specifically designed for sampling from high-dimensional
Gaussian-like distributions. Unlike gradient-based optimization, ESS does NOT require
gradients, making it more robust in non-smooth regions.

Critical fix: training-prior ellipse
-------------------------------------
The standard ESS auxiliary variable nu ~ N(0,I) defines ellipses in the
model prior space. But the oracle's training latents lie in a tight cluster
at x_mean ≈ [-0.83, 0.36, ...] with x_std ≈ [0.08-0.29] — 3-6σ away from
N(0,I) in every dimension. Using nu ~ N(0,I) draws ellipses that mostly
pass through OOD decoder regions → degenerate proteins → Rg ≈ 3-5 Å.

Fix: draw nu from N(x_mean, diag(x_std^2)) so ellipses stay within
the training distribution where the decoder produces valid proteins.

Reference:
  Murray, Iain M., Ryan P. Adams, and David JC MacKay.
  "Elliptical slice sampling."
  ICML, 2010.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

import torch


def run_ess(
    z_start: torch.Tensor,
    target_fn: torch.nn.Module,
    steps: int = 50,
    warm_start: bool = True,
    warm_steps: int = 20,
    warm_lr: float = 0.1,
    x_mean: Optional[torch.Tensor] = None,
    x_std:  Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, List[float]]:
    """
    Elliptical Slice Sampling (ESS) for latent space exploration.

    Training-prior fix
    ------------------
    Pass x_mean and x_std (from the oracle .npz) so that the auxiliary
    variable nu is drawn from N(x_mean, diag(x_std^2)) instead of N(0,I).
    This keeps all ellipse proposals within the oracle's training domain,
    where the decoder produces valid protein structures.
    If x_mean/x_std are None, falls back to N(0,I) (legacy behaviour).

    Args:
        z_start:    Initial latent vector [latent_dim]
        target_fn:  Oracle function that returns OracleOutput
        steps:      Number of MCMC steps (default 50)
        warm_start: Run gradient ascent to find a good starting point (default True)
        warm_steps: Number of gradient-ascent warm-start steps (default 20)
        warm_lr:    Learning rate for warm-start gradient ascent (default 0.1)
        x_mean:     Training-prior mean [latent_dim] from oracle checkpoint
        x_std:      Training-prior std  [latent_dim] from oracle checkpoint

    Returns:
        - z_final: Final sample after all steps [latent_dim]
        - trajectory: List of accepted log_likelihood values at each step
    """
    # Optional warm-start: short gradient ascent to escape the prior mean
    if warm_start:
        z_init = z_start.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z_init], lr=warm_lr)
        for _ in range(warm_steps):
            opt.zero_grad()
            loss = -target_fn(z_init).log_likelihood
            loss.backward()
            opt.step()
        z = z_init.detach()
    else:
        z = z_start.clone().detach()

    # Move prior params to same device as z
    if x_mean is not None:
        x_mean = x_mean.to(z.device)
        x_std  = x_std.to(z.device)

    trajectory: List[float] = []

    with torch.no_grad():
        for step in range(steps):
            # Step 1: Evaluate current likelihood
            oracle_output = target_fn(z)
            current_log_like = oracle_output.log_likelihood.item()

            # Step 2: Draw slice threshold
            u = random.uniform(0, 1)
            threshold_log_like = current_log_like + math.log(u)

            # Step 3: Draw auxiliary variable from TRAINING PRIOR
            # N(x_mean, diag(x_std^2)) instead of N(0,I)
            # This ensures ellipse proposals stay in the decoder's valid region
            eps = torch.randn_like(z)
            if x_mean is not None:
                nu = x_mean + eps * x_std
            else:
                nu = eps  # fallback: N(0,I) (legacy)

            # Step 4: Initialize the bracket
            theta = random.uniform(0, 2 * math.pi)
            theta_min = theta - 2 * math.pi
            theta_max = theta

            # Step 5: Search on the ellipse
            max_attempts = 100
            attempts = 0

            while attempts < max_attempts:
                attempts += 1
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                z_prime = z * cos_theta + nu * sin_theta

                oracle_output_prime = target_fn(z_prime)
                candidate_log_like = oracle_output_prime.log_likelihood.item()

                if candidate_log_like > threshold_log_like:
                    z = z_prime
                    trajectory.append(candidate_log_like)
                    break
                else:
                    if theta > 0:
                        theta_max = theta
                    else:
                        theta_min = theta
                    theta = random.uniform(theta_min, theta_max)

            if attempts == max_attempts:
                print(f"  [ESS] Warning: max_attempts reached at step {step}")
                trajectory.append(current_log_like)

            if (step + 1) % max(1, steps // 10) == 0 or step == 0:
                print(
                    f"  [ESS]  Step {step + 1:3d}/{steps} | "
                    f"log_score: {trajectory[-1]:8.4f} | "
                    f"||z||: {torch.norm(z).item():7.4f} | "
                    f"attempts: {attempts}"
                )

    return z, trajectory


if __name__ == "__main__":
    print("Testing run_ess...")

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

    print(f"\nRunning ESS (latent_dim={latent_dim})...")
    z_final, trajectory = run_ess(
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
