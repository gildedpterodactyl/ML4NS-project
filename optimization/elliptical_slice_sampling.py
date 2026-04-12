#!/usr/bin/env python3
"""
Algorithm 2: Elliptical Slice Sampling (ESS)

ESS is an MCMC algorithm specifically designed for sampling from high-dimensional
Gaussian-like distributions. Unlike gradient-based optimization, ESS does NOT require
gradients, making it more robust in non-smooth regions.

Key properties:
- Does NOT use PyTorch gradients (use torch.no_grad() for speed)
- Requires NO tuning of step size or learning rate
- Naturally handles high-dimensional spaces
- Produces a sequence of samples from the likelihood distribution

The algorithm works by:
1. Drawing an auxiliary "twin" variable nu ~ N(0, I)
2. Finding a slice threshold through log-likelihood
3. Searching an ellipse for points above the threshold
4. Accepting when a valid point is found

Warm-start note
---------------
When the target Rg is far from the oracle mean (intercept≈26.8 Å), all
points near z_start have similarly terrible scores and the chain freezes.
Pass warm_start=True (default) to run 20 gradient-ascent steps that move
z_start into a moderate-score region before the MCMC chain begins.  The
warm-start result is only used as the MCMC initial point; it is NOT
included in the returned trajectory.

Reference:
  Murray, Iain M., Ryan P. Adams, and David JC MacKay.
  "Elliptical slice sampling."
  ICML, 2010.
"""

from __future__ import annotations

import math
import random
from typing import List

import torch


def run_ess(
    z_start: torch.Tensor,
    target_fn: torch.nn.Module,
    steps: int = 50,
    warm_start: bool = True,
    warm_steps: int = 20,
    warm_lr: float = 0.1,
) -> tuple[torch.Tensor, List[float]]:
    """
    Elliptical Slice Sampling (ESS) for latent space exploration.

    This algorithm draws samples from a distribution proportional to:
        P(z) ∝ exp(log_likelihood(z))

    Unlike gradient ascent, ESS:
    - Requires NO gradient computation during MCMC (only during warm-start)
    - Requires NO learning rate tuning for the MCMC phase
    - Explores multiple high-likelihood regions
    - Produces an MCMC chain (samples, not just optimization)

    Algorithm:
        [Optional warm-start: 20 gradient-ascent steps to reach a
         moderate-score region before starting the chain]

        For each MCMC step:
        1. Evaluate current log_likelihood: L = log p(z)
        2. Draw slice threshold: L_slice = L + log(U), where U ~ Uniform(0,1)
        3. Draw auxiliary variable: nu ~ N(0, I)  [called the "twin"]
        4. Initialize bracket: theta ~ Uniform(0, 2π)
        5. Search on ellipse: z' = z*cos(θ) + nu*sin(θ)
           - If log_lik(z') > L_slice, accept z' and continue
           - Otherwise, shrink the bracket and try a new angle
        6. Accept z' as the next sample

    Args:
        z_start:    Initial latent vector [latent_dim]
        target_fn:  Oracle function that returns OracleOutput
        steps:      Number of MCMC steps (default 50)
        warm_start: Run gradient ascent to find a good starting point
                    before beginning MCMC (default True).  Strongly
                    recommended when the target is far from the prior mean.
        warm_steps: Number of gradient-ascent warm-start steps (default 20)
        warm_lr:    Learning rate for warm-start gradient ascent (default 0.1)

    Returns:
        - z_final: Final sample after all steps [latent_dim]
        - trajectory: List of accepted log_likelihood values at each step

    Notes:
        - The warm-start result is used only as z_0 for MCMC; it is NOT
          added to the trajectory.
        - Early MCMC samples may be influenced by z_0; consider discarding
          them (burn-in) before analysis.
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

    trajectory: List[float] = []

    with torch.no_grad():
        for step in range(steps):
            # Step 1: Evaluate current likelihood
            oracle_output = target_fn(z)
            current_log_like = oracle_output.log_likelihood.item()

            # Step 2: Draw slice threshold
            u = random.uniform(0, 1)
            threshold_log_like = current_log_like + math.log(u)

            # Step 3: Draw the auxiliary variable
            nu = torch.randn_like(z)

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
