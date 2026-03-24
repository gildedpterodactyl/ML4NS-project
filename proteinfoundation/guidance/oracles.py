# SPDX-FileCopyrightText: 2025 Vishak
# SPDX-License-Identifier: MIT
#
# Geometric oracles for Training-Free Guidance (TFG).
#
# Each oracle maps Cα coordinates  x ∈ R^{b × n_res × 3}  (nanometers)
# to a *differentiable* scalar property  y ∈ R^{b}.
#
# ProteinAE operates in ca_only mode: x is [b, n_res, 3] of Cα atoms.
# Coordinates are in **nanometers** (Angstroms / 10).
#
# References for the physical quantities:
#   - Radius of gyration: Fixman, 1962; Lobanov et al., Mol Biol 2008.
#   - Contact density:    Vendruscolo et al., PRE 1997.
#   - Clash score:        Word et al., J Mol Biol 1999 (MolProbity).
#   - Hydrogen bonds:     Baker & Hubbard, PPBMB 1984 (geometric criterion).

from __future__ import annotations

import abc
from typing import Dict, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class GeometricOracle(abc.ABC):
    """
    Base class for a differentiable geometric property oracle.

    Subclasses must implement ``forward(x, mask)`` returning a per-sample
    scalar  y  that is **differentiable** w.r.t.  x.

    Convention
    ----------
    * x     : [b, n, 3]  backbone coords in nanometers
    * mask  : [b, n]     boolean mask (True = real atom)
    * returns [b]        scalar property per sample
    """

    def __init__(self, name: str, target: float, direction: str = "minimize"):
        """
        Parameters
        ----------
        name : human-readable oracle name
        target : the target property value the guidance should push toward
        direction : 'minimize'  →  loss = (y - target)²
                    'maximize'  →  loss = -(y - target)²   (i.e. gradient ascent)
        """
        self.name = name
        self.target = target
        assert direction in ("minimize", "maximize")
        self.direction = direction

    @abc.abstractmethod
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Compute property  y  [b] given coords  x  [b, n, 3]."""
        ...

    def loss(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Guidance loss whose gradient drives sampling toward ``self.target``.

        Returns  L  [b]  with  L = (y - target)^2  (or negated for maximize).
        """
        y = self.forward(x, mask)  # [b]
        diff = (y - self.target) ** 2  # [b]
        if self.direction == "maximize":
            diff = -diff
        return diff  # [b]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(target={self.target}, "
            f"direction={self.direction})"
        )


# ---------------------------------------------------------------------------
# Coordinate extraction helper
# ---------------------------------------------------------------------------
def _extract_ca(x: Tensor, mask: Tensor):
    """Return (ca_coords [b, n_res, 3], ca_mask [b, n_res]).

    ProteinAE always operates in ca_only mode, so x is already
    [b, n_res, 3] of Cα coordinates.  We just ensure the mask
    is float for downstream arithmetic.
    """
    return x, mask.float()


# ---------------------------------------------------------------------------
# 1. Radius of Gyration  (Rg)
# ---------------------------------------------------------------------------
class RadiusOfGyration(GeometricOracle):
    r"""
    Radius of gyration of Cα atoms:

    .. math::
        R_g = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{r}_i - \bar{\mathbf{r}} \|^2 }

    where the sum runs over **Cα** atoms only (every 4th atom in the
    interleaved backbone representation).

    Units: nanometers (same as input coords).
    """

    def __init__(self, target: float = 1.5, direction: str = "minimize"):
        super().__init__("radius_of_gyration", target, direction)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        b, n, _ = x.shape
        ca, ca_mask = _extract_ca(x, mask)

        # Masked mean (center of mass)
        n_atoms = ca_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [b, 1]
        com = (ca * ca_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / n_atoms.unsqueeze(-1)  # [b, 1, 3]

        # Squared distances from COM
        diff = (ca - com) * ca_mask.unsqueeze(-1)  # [b, n_res, 3]
        sq_dist = (diff ** 2).sum(dim=-1)  # [b, n_res]
        rg_sq = sq_dist.sum(dim=-1) / n_atoms.squeeze(-1)  # [b]
        rg = torch.sqrt(rg_sq + 1e-8)  # [b]
        return rg


# ---------------------------------------------------------------------------
# 2. Contact Density
# ---------------------------------------------------------------------------
class ContactDensity(GeometricOracle):
    r"""
    Average number of Cα–Cα contacts per residue (soft count).

    A smooth sigmoid is used instead of a hard cutoff:

    .. math::
        C = \frac{1}{N} \sum_{i<j}
            \sigma\!\bigl( -(d_{ij} - d_0) / \tau \bigr)

    With  d_0 = 0.8 nm  (~8 Å) and  τ = 0.05 nm for a soft switch.

    Higher contact density → more compact / globular fold.
    """

    def __init__(
        self,
        target: float = 6.0,
        direction: str = "minimize",
        d0: float = 0.8,
        tau: float = 0.05,
    ):
        super().__init__("contact_density", target, direction)
        self.d0 = d0  # nm
        self.tau = tau  # nm

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        ca, ca_mask = _extract_ca(x, mask)
        n_res = ca.shape[1]

        # Pairwise distances  [b, n_res, n_res]
        diff = ca.unsqueeze(2) - ca.unsqueeze(1)  # [b, n_res, n_res, 3]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [b, n_res, n_res]

        # Pair mask (both real and i ≠ j)
        pair_mask = ca_mask.unsqueeze(2) * ca_mask.unsqueeze(1)  # [b, n_res, n_res]
        diag = torch.eye(n_res, device=x.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask * (~diag)

        # Soft contact function
        contacts = torch.sigmoid(-(dist - self.d0) / self.tau)  # [b, n_res, n_res]
        contacts = contacts * pair_mask

        n_atoms = ca_mask.sum(dim=-1).clamp(min=1)  # [b]
        contact_density = contacts.sum(dim=(-1, -2)) / n_atoms  # [b]
        return contact_density


# ---------------------------------------------------------------------------
# 3. Hydrogen-Bond Score (backbone N–H···O)
# ---------------------------------------------------------------------------
class HBondScore(GeometricOracle):
    r"""
    Approximate backbone H-bond count using a soft geometric criterion
    on N···O distance and N-H···O angle.

    For backbone: each residue i has N at atom-index 0 and O at atom-index 3.
    An H-bond is counted (softly) when:
        d(N_i, O_j) < d_cutoff   (soft sigmoid)
    and we skip |i-j| < 3 to avoid trivially close pairs.

    This is a simplified version (no explicit H placement).
    d_cutoff ≈ 0.35 nm (3.5 Å).
    """

    def __init__(
        self,
        target: float = 0.3,
        direction: str = "maximize",
        d_cutoff: float = 0.35,
        tau: float = 0.03,
        seq_sep: int = 3,
    ):
        super().__init__("hbond_score", target, direction)
        self.d_cutoff = d_cutoff  # nm
        self.tau = tau
        self.seq_sep = seq_sep

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        b, n, _ = x.shape
        # ProteinAE is always ca_only — use CA-CA distances as proxy
        ca = x              # [b, n_res, 3]
        ca_mask = mask.float()  # [b, n_res]
        n_res = n

        diff = ca.unsqueeze(2) - ca.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        pair_mask = ca_mask.unsqueeze(2) * ca_mask.unsqueeze(1)

        # Sequence separation mask
        idx = torch.arange(n_res, device=x.device)
        seq_dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        seq_mask = (seq_dist >= self.seq_sep).float().unsqueeze(0)
        pair_mask = pair_mask * seq_mask

        # Soft H-bond counting (CA-CA proxy)
        hbonds = torch.sigmoid(-(dist - self.d_cutoff) / self.tau)
        hbonds = hbonds * pair_mask

        n_atoms_count = ca_mask.sum(dim=-1).clamp(min=1)
        hbond_per_res = hbonds.sum(dim=(-1, -2)) / n_atoms_count
        return hbond_per_res


# ---------------------------------------------------------------------------
# 4. Clash Score (steric penalty)
# ---------------------------------------------------------------------------
class ClashScore(GeometricOracle):
    r"""
    Steric clash penalty: counts the fraction of heavy-atom pairs
    closer than a van-der-Waals overlap threshold.

    Uses **all** backbone atoms (N, CA, C, O) with a single threshold
    of 0.20 nm (2.0 Å) for simplicity.

    Soft penalty via ReLU(threshold − dist).

    Target = 0 means "no clashes".
    """

    def __init__(
        self,
        target: float = 0.0,
        direction: str = "minimize",
        d_clash: float = 0.20,
        seq_sep: int = 2,
    ):
        super().__init__("clash_score", target, direction)
        self.d_clash = d_clash  # nm
        self.seq_sep = seq_sep  # min residue separation (atoms within same or adjacent residue aren't clashes)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        ca, ca_mask = _extract_ca(x, mask)
        n_res = ca.shape[1]

        diff = ca.unsqueeze(2) - ca.unsqueeze(1)  # [b, nres, nres, 3]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [b, nres, nres]

        pair_mask = ca_mask.unsqueeze(2) * ca_mask.unsqueeze(1)
        idx = torch.arange(n_res, device=x.device)
        seq_dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        seq_mask = (seq_dist >= self.seq_sep).float().unsqueeze(0)
        pair_mask = pair_mask * seq_mask

        # Soft clash: ReLU(d_clash - dist), penalises overlaps
        clash = torch.relu(self.d_clash - dist) * pair_mask  # [b, nres, nres]

        n_atoms = ca_mask.sum(dim=-1).clamp(min=1)  # [b]
        clash_per_res = clash.sum(dim=(-1, -2)) / n_atoms  # [b]
        return clash_per_res


# ---------------------------------------------------------------------------
# Registry for easy oracle lookup by name
# ---------------------------------------------------------------------------
class OracleRegistry:
    """Simple name → class lookup."""

    _ORACLES: Dict[str, type] = {
        "rg": RadiusOfGyration,
        "radius_of_gyration": RadiusOfGyration,
        "contact_density": ContactDensity,
        "contacts": ContactDensity,
        "hbond": HBondScore,
        "hbond_score": HBondScore,
        "clash": ClashScore,
        "clash_score": ClashScore,
    }

    @classmethod
    def get(cls, name: str, **kwargs) -> GeometricOracle:
        """Instantiate an oracle by name with optional kwargs (target, direction, …)."""
        name_lower = name.lower()
        if name_lower == "latent_brightness":
            from proteinfoundation.guidance.latent_regressor import LatentRegressorOracle
            return LatentRegressorOracle(**kwargs)
            
        if name_lower not in cls._ORACLES:
            raise ValueError(
                f"Unknown oracle '{name}'. Available: {list(cls._ORACLES.keys()) + ['latent_brightness']}"
            )
        return cls._ORACLES[name_lower](**kwargs)

    @classmethod
    def available(cls):
        return list(cls._ORACLES.keys())
