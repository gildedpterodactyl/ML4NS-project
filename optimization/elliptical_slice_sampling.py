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

This naturally explores the space more thoroughly than pure gradient ascent,
discovering multiple high-scoring regions.

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
) -> tuple[torch.Tensor, List[float]]:
    """
    Elliptical Slice Sampling (ESS) for latent space exploration.

    This algorithm draws samples from a distribution proportional to:
        P(z) ∝ exp(log_likelihood(z))

    Unlike gradient ascent, ESS:
    - Requires NO gradient computation
    - Requires NO learning rate tuning
    - Explores multiple high-likelihood regions
    - Produces an MCMC chain (samples, not just optimization)

    Algorithm:
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
        z_start: Initial latent vector [latent_dim]
        target_fn: Oracle function that returns OracleOutput
                  IMPORTANT: evaluated with torch.no_grad() for efficiency
        steps: Number of MCMC steps (default 50)

    Returns:
        - z_final: Final sample after all steps [latent_dim]
        - trajectory: List of accepted log_likelihood values at each step
                     Useful for diagnosing MCMC convergence and mixing

    Notes:
        - Early samples may be influenced by z_start; consider discarding them
          (burn-in) before analysis
        - All samples in trajectory are accepted (no rejection sampling),
          making it very efficient
        - The algorithm is "slice sampling" because it samples from a slice
          of the likelihood landscape

    Example:
        >>> from optimization.target_function import TargetFunction
        >>> target_fn = TargetFunction.from_checkpoint("model.npz", target_value=65.0)
        >>> z_start = torch.randn(256)
        >>> z_final, scores = run_ess(z_start, target_fn, steps=100)
        >>> print(f"Mean score (last 50 steps): {sum(scores[-50:])/50:.4f}")
    """
    # Use no_grad for efficiency (ESS doesn't use gradients)
    z = z_start.clone().detach()
    trajectory: List[float] = []

    with torch.no_grad():
        for step in range(steps):
            # Step 1: Evaluate current likelihood
            oracle_output = target_fn(z)
            current_log_like = oracle_output.log_likelihood.item()

            # Step 2: Draw slice threshold (this creates the "slice")
            # L_slice = L + log(U) where U ~ Uniform(0,1)
            # This is equivalent to L - Exponential(1)
            u = random.uniform(0, 1)
            threshold_log_like = current_log_like + math.log(u)

            # Step 3: Draw the auxiliary variable (the "twin")
            # This is what makes ESS work in high dimensions
            nu = torch.randn_like(z)

            # Step 4: Initialize the bracket for the angle search
            # theta is uniformly sampled from [0, 2π]
            theta = random.uniform(0, 2 * math.pi)
            theta_min = theta - 2 * math.pi
            theta_max = theta

            # Step 5: The inner loop: search on the ellipse
            # The ellipse is parameterized as: z' = z*cos(θ) + nu*sin(θ)
            max_attempts = 100  # Prevent infinite loops (very rare)
            attempts = 0

            while attempts < max_attempts:
                attempts += 1

                # Candidate point on the ellipse
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                z_prime = z * cos_theta + nu * sin_theta

                # Evaluate candidate
                oracle_output_prime = target_fn(z_prime)
                candidate_log_like = oracle_output_prime.log_likelihood.item()

                # Check: Is the candidate above the slice?
                if candidate_log_like > threshold_log_like:
                    # ACCEPT: This point is above the slice
                    z = z_prime
                    trajectory.append(candidate_log_like)
                    break
                else:
                    # REJECT: Shrink the bracket and try again
                    if theta > 0:
                        theta_max = theta
                    else:
                        theta_min = theta

                    # Draw a new angle from the shrunk bracket
                    theta = random.uniform(theta_min, theta_max)

            if attempts == max_attempts:
                # Safety fallback (shouldn't happen with reasonable likelihood)
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
    # Quick test
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
