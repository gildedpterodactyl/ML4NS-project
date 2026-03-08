# Training-Free Guidance (TFG) with Geometric Oracles
# for ProteinAE flow-matching generation.
#
# References:
#   [1] Zheng et al., "A Training-Free Conditional Diffusion Model for
#       Molecular Property Guidance", NeurIPS 2024 Workshop.
#   [2] Song et al., "Loss-Guided Diffusion Models for Plug-and-Play
#       Controllable Generation", ICML 2023.
#
# All oracles operate on backbone coordinates in NANOMETERS (internal
# ProteinAE convention: Angstroms / 10).

from proteinfoundation.guidance.oracles import (
    GeometricOracle,
    RadiusOfGyration,
    ContactDensity,
    HBondScore,
    ClashScore,
    OracleRegistry,
)

__all__ = [
    "GeometricOracle",
    "RadiusOfGyration",
    "ContactDensity",
    "HBondScore",
    "ClashScore",
    "OracleRegistry",
]
