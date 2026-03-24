# SPDX-FileCopyrightText: 2025 Vishak
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from proteinfoundation.guidance.oracles import GeometricOracle

class MLP(nn.Module):
    """
    The regression MLP used to predict a scalar property from 
    mean-pooled latent representations (z ∈ R^8).
    """
    def __init__(self, input_dim=8, hidden_dim1=256, hidden_dim2=128, output_dim=1):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class LatentRegressorOracle(GeometricOracle):
    """
    Guides the Latent space of the diffusion model directly using an MLP 
    trained to predict biological properties (Brightness, Thermostability, etc.).
    
    x refers to latent variables here (z ∈ R^{b × L × 8}) instead of 3D coords.
    """
    def __init__(
        self, 
        model_path: str,
        target: float, 
        direction: str = "maximize",
        input_dim: int = 8,
    ):
        super().__init__("latent_regressor", target, direction)
        
        # Load the regresson model
        self.mlp = MLP(input_dim=input_dim)
        
        # Load state dictionary
        state_dict = torch.load(model_path, map_location="cpu")
        self.mlp.load_state_dict(state_dict)
        
        # Freeze parameters
        self.mlp.eval()
        for param in self.mlp.parameters():
            param.requires_grad = False
            
    def set_device(self, device):
        self.mlp = self.mlp.to(device)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Compute scalar property via the regression model.
        
        Parameters
        ----------
        x : Tensor, shape [b, L, 8]
            Latent representations.
        mask : Tensor, shape [b, L]
            Boolean mask representing valid sequence padding.
            
        Returns
        -------
        output : Tensor, shape [b] 
            The predicted scalar properties.
        """
        # x is [b, L, 8], mask is [b, L]
        # We need to length-wise mean pool the sequence using the mask.
        b, L, d = x.shape
        
        # Apply the mask: multiply invalid positions by 0
        mask_f = mask.float()
        x_masked = x * mask_f.unsqueeze(-1)
        
        # Compute mean representation (sum over L divided by non-zero elements)
        non_pad_lengths = mask_f.sum(dim=1).clamp(min=1)  # [b]
        z_pooled = x_masked.sum(dim=1) / non_pad_lengths.unsqueeze(-1)  # [b, 8]
        
        # Ensure MLP runs on same device as x
        if self.mlp.layer1.weight.device != x.device:
            self.set_device(x.device)
            
        # Predict the property
        output = self.mlp(z_pooled).squeeze(-1)  # [b]
        return output
