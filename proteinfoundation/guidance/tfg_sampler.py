# SPDX-FileCopyrightText: 2025 Vishak
# SPDX-License-Identifier: MIT
#
# Training-Free Guidance (TFG) utilities for flow-matching sampling.
#
# The core idea: at each ODE step we have the *predicted clean sample*
# x̂₁.  We compute ∇_x oracle_loss(x̂₁) and inject it into the
# velocity field:
#
#   v_guided = v_model − η · ∇_x L(x̂₁)
#
# where η is the guidance scale and L is the oracle loss (pushes the
# property toward the target).
#
# The gradient is taken w.r.t. the *noisy* sample x_t (through the
# prediction x̂₁ which depends on x_t), following [Zheng et al., 2024;
# Song et al., 2023].

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from proteinfoundation.guidance.oracles import GeometricOracle


class CompositeOracle:
    """
    Combines multiple geometric oracles into one guidance signal.

    The composite loss is a weighted sum:
        L_total = Σ_k  w_k · L_k(x)
    where  L_k  is oracle_k.loss(x, mask).
    """

    def __init__(
        self,
        oracles: List[GeometricOracle],
        weights: Optional[List[float]] = None,
    ):
        self.oracles = oracles
        self.weights = weights or [1.0] * len(oracles)
        assert len(self.weights) == len(self.oracles)

    def loss(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Compute composite loss  [b].

        Parameters
        ----------
        x    : [b, n, 3]  backbone coords in nm
        mask : [b, n]     boolean mask
        """
        total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for oracle, w in zip(self.oracles, self.weights):
            total = total + w * oracle.loss(x, mask)
        return total  # [b]

    def __repr__(self) -> str:
        parts = [f"{w}×{o}" for o, w in zip(self.oracles, self.weights)]
        return f"CompositeOracle([{', '.join(parts)}])"


def compute_guidance_gradient(
    x_1_pred: Tensor,
    coords_mask: Tensor,
    oracle: GeometricOracle | CompositeOracle,
    guidance_scale: float = 1.0,
) -> Tensor:
    """
    Compute the TFG guidance gradient to steer the velocity field.

    Takes ∂L/∂x̂₁ — the gradient of the oracle loss w.r.t. the
    predicted clean sample.  This gradient is then used to modify
    the velocity:  v_guided = v − η · ∂L/∂x̂₁.

    Parameters
    ----------
    x_1_pred     : [b, n, 3]  predicted clean sample from the network
    coords_mask  : [b, n]     boolean mask
    oracle       : the oracle (or CompositeOracle) providing .loss()
    guidance_scale : η multiplier

    Returns
    -------
    grad : [b, n, 3]  gradient to *subtract* from velocity
           (i.e.  v_guided = v − grad)
    """
    if guidance_scale == 0.0:
        return torch.zeros_like(x_1_pred)

    # Lightning's predict_step runs under torch.inference_mode(), which
    # prevents ANY autograd graph construction — even with enable_grad().
    # We must exit inference_mode AND enable grad to compute oracle grads.
    with torch.inference_mode(False):
        with torch.enable_grad():
            # Clone + requires_grad INSIDE the non-inference context
            x_1 = x_1_pred.detach().clone().float().requires_grad_(True)
            mask_float = coords_mask.detach().float()

            loss = oracle.loss(x_1, mask_float)  # [b]
            loss_sum = loss.sum()

            grad = torch.autograd.grad(
                loss_sum,
                x_1,
                create_graph=False,
                retain_graph=False,
            )[0]  # [b, n, 3]

    # Mask and scale
    grad = grad * coords_mask.unsqueeze(-1).float()  # zero out padding
    return guidance_scale * grad  # [b, n, 3]
