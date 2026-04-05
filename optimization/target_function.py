#!/usr/bin/env python3
"""
Differentiable Oracle Wrapper for Latent Space Optimization.

This module implements TargetFunction(nn.Module), which wraps a pre-trained
Oracle neural network (TM predictor) and provides a differentiable interface
for gradient-based optimization algorithms.

The Oracle is trained using ridge regression on ProteinAE latent embeddings
and saved as a .npz file. We wrap it in PyTorch to enable gradient flow.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn


class OracleOutput(NamedTuple):
    """Output structure for Oracle predictions."""
    pred_value: torch.Tensor  # Predicted target value (e.g., Tm)
    likelihood: torch.Tensor  # Score = exp(-(Pred - Target)^2)
    log_likelihood: torch.Tensor  # log_score = -(Pred - Target)^2


class RidgeRegressor(nn.Module):
    """
    PyTorch wrapper for Ridge regression model.
    
    Loads pre-trained coefficients and intercept from an .npz file
    and provides a differentiable forward pass.
    """

    def __init__(self, coef: np.ndarray, intercept: float):
        """
        Args:
            coef: Ridge regression coefficients [n_features]
            intercept: Ridge regression intercept (scalar)
        """
        super().__init__()
        self.register_buffer(
            "coef",
            torch.from_numpy(coef).float().unsqueeze(0),  # [1, n_features]
        )
        self.register_buffer(
            "intercept",
            torch.tensor([intercept], dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Linear prediction: y = x @ coef.T + intercept
        
        Args:
            x: Input tensor [batch, n_features]

        Returns:
            Predictions [batch]
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return (x @ self.coef.T).squeeze(-1) + self.intercept


class TargetFunction(nn.Module):
    r"""
    Differentiable Oracle Function for Latent Space Optimization.

    This module wraps a pre-trained TM (or other target property) predictor
    and provides a likelihood-based scoring interface suitable for both
    gradient-based and MCMC optimization algorithms.

    The likelihood score is defined as:
        Score = exp(-(Pred(z) - Target)^2)

    And the log-likelihood:
        log_Score = -(Pred(z) - Target)^2

    This formulation:
    - Always returns positive scores (0 to 1]
    - Peaks at 1.0 when Pred(z) == Target
    - Prefers working with log-scale to prevent numerical underflow

    Example:
        >>> target_fn = TargetFunction.from_checkpoint(
        ...     model_path="regression/outputs/ridge_latent_tm_model.npz",
        ...     target_value=65.0,  # Tm=65°C
        ...     x_mean=np.zeros(256),
        ...     x_std=np.ones(256)
        ... )
        >>> z = torch.randn(256)
        >>> output = target_fn(z)
        >>> log_score = output.log_likelihood
        >>> print(f"Log-likelihood: {log_score.item():.4f}")
    """

    def __init__(
        self,
        oracle: RidgeRegressor,
        target_value: float,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        temperature: float = 1.0,
    ):
        """
        Initialize the TargetFunction.

        Args:
            oracle: RidgeRegressor module wrapping pre-trained coefficients
            target_value: Target value (e.g., desired Tm in °C) to optimize towards
            x_mean: Standardization mean [n_features] from training
            x_std: Standardization std [n_features] from training
            temperature: Temperature parameter for likelihood scaling (default 1.0)
                        Higher values flatten the likelihood landscape.
        """
        super().__init__()
        self.oracle = oracle
        self.target_value = target_value
        self.temperature = temperature

        self.register_buffer("_x_mean", torch.from_numpy(x_mean).float())
        self.register_buffer("_x_std", torch.from_numpy(x_std).float())

    def forward(self, z: torch.Tensor) -> OracleOutput:
        """
        Evaluate the likelihood of a latent vector.

        Args:
            z: Latent vector [n_features] or batch [batch_size, n_features]
               Can optionally have requires_grad=True for gradient-based optimization.

        Returns:
            OracleOutput with:
                - pred_value: Predicted target property
                - likelihood: exp(-(Pred - Target)^2 / temperature)
                - log_likelihood: -(Pred - Target)^2 / temperature
        """
        # Ensure 2D for batch processing
        was_1d = z.ndim == 1
        if was_1d:
            z = z.unsqueeze(0)

        # Standardize using training statistics
        z_standardized = (z - self._x_mean) / self._x_std

        # Predict target value (e.g., Tm)
        pred = self.oracle(z_standardized)
        if was_1d:
            pred = pred.squeeze(0)

        # Calculate likelihood
        squared_error = (pred - self.target_value) ** 2
        log_likelihood = -squared_error / self.temperature
        likelihood = torch.exp(log_likelihood)

        return OracleOutput(
            pred_value=pred,
            likelihood=likelihood,
            log_likelihood=log_likelihood,
        )

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | Path,
        target_value: float,
        temperature: float = 1.0,
    ) -> TargetFunction:
        """
        Load a TargetFunction from a trained Ridge regression checkpoint.

        Args:
            model_path: Path to ridge_latent_tm_model.npz saved by train_tm_from_latent.py
            target_value: Target value to optimize towards (e.g., Tm=65)
            temperature: Likelihood temperature parameter (default 1.0)

        Returns:
            Fully initialized TargetFunction ready for optimization

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint missing required keys
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        with np.load(model_path) as npz:
            required_keys = {"coef", "intercept", "x_mean", "x_std"}
            missing = required_keys - set(npz.files)
            if missing:
                raise KeyError(f"Checkpoint missing keys: {missing}")

            coef = npz["coef"].astype(np.float32)
            intercept = float(npz["intercept"].flat[0])
            x_mean = npz["x_mean"].astype(np.float32)
            x_std = npz["x_std"].astype(np.float32)

        oracle = RidgeRegressor(coef, intercept)
        return cls(
            oracle=oracle,
            target_value=target_value,
            x_mean=x_mean,
            x_std=x_std,
            temperature=temperature,
        )

    def get_latent_dim(self) -> int:
        """Get the dimensionality of the expected input latent space."""
        return self._x_mean.shape[0]

    def get_target_value(self) -> float:
        """Get the target value this function is optimizing towards."""
        return self.target_value


if __name__ == "__main__":
    # Quick test: Load the model and evaluate a random vector
    import sys

    checkpoint_path = Path("regression/outputs/ridge_latent_tm_model.npz")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print("Loading TargetFunction from checkpoint...")
    target_fn = TargetFunction.from_checkpoint(
        model_path=checkpoint_path,
        target_value=65.0,  # Target Tm = 65°C
        temperature=1.0,
    )

    latent_dim = target_fn.get_latent_dim()
    print(f"Latent dimension: {latent_dim}")

    # Test forward pass
    z_test = torch.randn(latent_dim)
    output = target_fn(z_test)

    print(f"\nTest forward pass:")
    print(f"  Input shape: {z_test.shape}")
    print(f"  Predicted Tm: {output.pred_value.item():.2f}°C")
    print(f"  Likelihood: {output.likelihood.item():.4f}")
    print(f"  Log-likelihood: {output.log_likelihood.item():.4f}")

    # Test with requires_grad for optimization
    z_opt = torch.randn(latent_dim, requires_grad=True)
    output_opt = target_fn(z_opt)
    print(f"\nWith requires_grad=True:")
    print(f"  Log-likelihood requires_grad: {output_opt.log_likelihood.requires_grad}")

    # Test backward pass
    loss = -output_opt.log_likelihood  # Minimize negative log-likelihood
    loss.backward()
    print(f"  Gradients computed: {z_opt.grad is not None}")
    print(f"  Gradient norm: {z_opt.grad.norm().item():.4f}")
