#!/usr/bin/env python3
"""
Algorithm 1: Penalized Gradient Ascent

This is the baseline optimization algorithm. It uses standard PyTorch optimization
with an explicit L2 penalty to prevent the latent vector from diverging.

Key differences from standard gradient descent:
- We MAXIMIZE the likelihood score (not minimize)
- We impose an L2 penalty to keep ||z|| bounded
- The loss is: -(log_likelihood - penalty_weight * ||z||^2)

The negative sign is critical: PyTorch optimizers implement gradient DESCENT,
but we want ASCENT (maximization). So we negate the objective.
"""

from __future__ import annotations

from typing import List

import torch


def run_gradient_ascent(
    z_start: torch.Tensor,
    target_fn: torch.nn.Module,
    steps: int = 50,
    lr: float = 0.1,
    penalty_weight: float = 0.05,
) -> tuple[torch.Tensor, List[float]]:
    """
    Penalized Gradient Ascent for latent space optimization.

    This algorithm iteratively updates the latent vector z to maximize:
        Objective(z) = log_likelihood(z) - penalty_weight * ||z||^2

    The penalty term keeps the latent vector from diverging into regions
    where the encoder/decoder may not work well.

    Algorithm:
        1. Initialize z from z_start with requires_grad=True
        2. Use Adam optimizer with learning rate lr
        3. For each step:
            - Compute log_score = target_fn(z).log_likelihood
            - Compute l2_penalty = penalty_weight * ||z||^2
            - Compute loss = -(log_score - l2_penalty)  [negative for descent]
            - Backpropagate and update z

    Args:
        z_start: Initial latent vector [latent_dim]
        target_fn: Oracle function (TargetFunction module) that returns OracleOutput
        steps: Number of optimization steps (default 50)
        lr: Adam learning rate (default 0.1)
        penalty_weight: L2 penalty coefficient (default 0.05)
                       Higher values keep z closer to origin.

    Returns:
        - z_final: Optimized latent vector [latent_dim]
        - trajectory: List of log_likelihood values at each step
                     Can be used to plot convergence curves

    Example:
        >>> from optimization.target_function import TargetFunction
        >>> target_fn = TargetFunction.from_checkpoint("model.npz", target_value=65.0)
        >>> z_start = torch.randn(256)
        >>> z_final, scores = run_gradient_ascent(z_start, target_fn, steps=100, lr=0.15)
        >>> print(f"Final score: {scores[-1]:.4f}")
    """
    # Clone and set up for gradient computation
    z = z_start.clone().detach().requires_grad_(True)

    # Adam optimizer
    optimizer = torch.optim.Adam([z], lr=lr)

    # Track log-likelihood trajectory
    trajectory: List[float] = []

    for step in range(steps):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Evaluate Oracle
        oracle_output = target_fn(z)
        log_score = oracle_output.log_likelihood

        # L2 penalty to prevent divergence
        l2_penalty = penalty_weight * torch.sum(z ** 2)

        # Objective: maximize (log_score - l2_penalty)
        # Since PyTorch minimizes loss, we negate:
        objective = log_score - l2_penalty
        loss = -objective

        # Backpropagation
        loss.backward()

        # Gradient step
        optimizer.step()

        # Log for monitoring
        trajectory.append(log_score.item())

        if (step + 1) % max(1, steps // 10) == 0 or step == 0:
            print(
                f"  [Gradient] Step {step + 1:3d}/{steps} | "
                f"log_score: {log_score.item():8.4f} | "
                f"||z||: {torch.norm(z).item():7.4f}"
            )

    return z.detach(), trajectory


if __name__ == "__main__":
    # Quick test
    print("Testing run_gradient_ascent...")

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

    print(f"\nRunning gradient ascent (latent_dim={latent_dim})...")
    z_final, trajectory = run_gradient_ascent(
        z_start=z_start,
        target_fn=target_fn,
        steps=50,
        lr=0.1,
        penalty_weight=0.05,
    )

    print(f"\nResults:")
    print(f"  Initial log_score: {trajectory[0]:.4f}")
    print(f"  Final log_score:   {trajectory[-1]:.4f}")
    print(f"  Improvement:       {trajectory[-1] - trajectory[0]:.4f}")
    print(f"  ||z_final||:       {torch.norm(z_final).item():.4f}")
