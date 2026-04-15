"""
ESS and TESS optimization in latent space for protein fitness optimization.

ESS  - Evolutionary Strategy Sampling: Gaussian perturbation in latent space,
       select top-k by predicted fitness.
TESS - Trustworthy ESS: ESS with a trust-region constraint that keeps candidates
       within a bounded Mahalanobis/L2 distance from the wild-type latent vector.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ESSConfig:
    """Configuration for ESS optimization."""
    population_size: int = 128          # Number of candidates per generation
    num_generations: int = 50           # Number of optimization steps
    sigma: float = 1.0                  # Initial perturbation scale (std dev)
    sigma_decay: float = 0.99           # Multiplicative decay of sigma per generation
    sigma_min: float = 0.1              # Minimum sigma
    top_k: int = 32                     # Elite candidates selected each generation
    num_restarts: int = 1               # Independent restarts (best is kept)
    seed: Optional[int] = None


@dataclass
class TESSConfig(ESSConfig):
    """Configuration for TESS (trust-region ESS) optimization."""
    trust_radius: float = 4.0           # Max L2 distance from WT latent (trust region)
    adaptive_trust: bool = True         # Expand/contract trust radius based on improvement
    trust_expand: float = 1.1           # Factor to expand trust radius on improvement
    trust_contract: float = 0.9         # Factor to contract trust radius on no improvement
    trust_min: float = 1.0              # Minimum trust radius
    trust_max: float = 10.0             # Maximum trust radius


def _select_elites(
    latents: torch.Tensor,
    fitness: torch.Tensor,
    top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-k candidates by predicted fitness (higher is better)."""
    topk_vals, topk_idx = torch.topk(fitness.squeeze(-1), k=min(top_k, len(fitness)))
    return latents[topk_idx], topk_vals


def _project_to_trust_region(
    z_perturbed: torch.Tensor,
    z_wt: torch.Tensor,
    trust_radius: float
) -> torch.Tensor:
    """
    Project candidates back into trust region (L2 ball around WT latent).

    Args:
        z_perturbed: Candidate latents [N, latent_dim]
        z_wt:        Wild-type latent  [1, latent_dim] or [latent_dim]
        trust_radius: Maximum L2 distance from z_wt

    Returns:
        z_projected: Projected latents [N, latent_dim]
    """
    if z_wt.dim() == 1:
        z_wt = z_wt.unsqueeze(0)

    delta = z_perturbed - z_wt                          # [N, D]
    dist = torch.norm(delta, dim=-1, keepdim=True)      # [N, 1]

    # Scale back to trust_radius where violated
    scale = torch.clamp(trust_radius / (dist + 1e-8), max=1.0)
    z_projected = z_wt + delta * scale

    return z_projected


def run_ess(
    model,
    wt_seqs: List[str],
    config: ESSConfig,
    device: Union[str, torch.device] = "cuda",
    verbose: bool = True,
) -> Tuple[List[str], List[float], torch.Tensor]:
    """
    ESS: Evolutionary Strategy Sampling in latent space.

    Algorithm:
        1. Encode WT sequences -> z_wt (seed population)
        2. For each generation:
            a. Perturb elite latents with Gaussian noise (scale=sigma)
            b. Predict fitness for all candidates
            c. Select top-k as new elites
            d. Decay sigma
        3. Decode best latents -> sequences

    Args:
        model:    Trained VAE model (BaseVAE subclass, in eval mode)
        wt_seqs:  Wild-type sequence(s) as starting point
        config:   ESS hyperparameters
        device:   Torch device
        verbose:  Print progress

    Returns:
        sequences:    Decoded protein sequences (sorted by fitness, best first)
        fitnesses:    Predicted fitness scores
        best_latents: Latent vectors of returned sequences [N, D]
    """
    model.eval()
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    best_overall_seqs = []
    best_overall_fits = []
    best_overall_z = None

    for restart in range(config.num_restarts):
        if verbose and config.num_restarts > 1:
            print(f"  [ESS] Restart {restart+1}/{config.num_restarts}")

        with torch.no_grad():
            # Step 1: Encode WT -> seed latent
            z_wt, mu_wt, _ = model.encode(wt_seqs)          # [B, D]
            z_seed = mu_wt.mean(dim=0, keepdim=True)         # [1, D] - mean of WT batch

        sigma = config.sigma
        # Seed population: broadcast + initial perturbation
        elites = z_seed.expand(config.top_k, -1).clone()    # [top_k, D]

        gen_best_fit = -float("inf")

        for gen in range(config.num_generations):
            with torch.no_grad():
                # Step 2a: Build population by perturbing elites
                repeats = max(1, config.population_size // config.top_k)
                z_pop = elites.repeat_interleave(repeats, dim=0)         # [P, D]
                noise = torch.randn_like(z_pop) * sigma
                z_pop = z_pop + noise                                    # [P, D]

                # Step 2b: Predict fitness
                fitness = model.predict(z_pop)                           # [P, 1]

                # Step 2c: Select top-k elites
                elites, elite_fits = _select_elites(z_pop, fitness, config.top_k)

                gen_best = elite_fits[0].item()
                if gen_best > gen_best_fit:
                    gen_best_fit = gen_best

            # Step 2d: Decay sigma
            sigma = max(config.sigma_min, sigma * config.sigma_decay)

            if verbose and (gen + 1) % 10 == 0:
                print(f"  [ESS] Gen {gen+1:3d}/{config.num_generations} | "
                      f"best_fit={gen_best_fit:.4f} | sigma={sigma:.4f}")

        # Decode elite latents -> sequences
        with torch.no_grad():
            seqs = model.generate_from_latent(elites)
            fits = model.predict(elites).squeeze(-1).cpu().tolist()

        # Sort by fitness descending
        sorted_pairs = sorted(zip(seqs, fits, list(range(len(seqs)))),
                               key=lambda x: -x[1])
        seqs_sorted = [p[0] for p in sorted_pairs]
        fits_sorted = [p[1] for p in sorted_pairs]
        idx_sorted  = [p[2] for p in sorted_pairs]
        z_sorted    = elites[idx_sorted]

        if not best_overall_seqs or fits_sorted[0] > best_overall_fits[0]:
            best_overall_seqs = seqs_sorted
            best_overall_fits = fits_sorted
            best_overall_z    = z_sorted

    return best_overall_seqs, best_overall_fits, best_overall_z


def run_tess(
    model,
    wt_seqs: List[str],
    config: TESSConfig,
    device: Union[str, torch.device] = "cuda",
    verbose: bool = True,
) -> Tuple[List[str], List[float], torch.Tensor]:
    """
    TESS: Trustworthy ESS - ESS with a trust-region constraint.

    All ESS steps apply, PLUS after perturbation:
        - Candidates violating the trust region (||z - z_wt|| > R) are
          projected back onto the L2 ball boundary.
        - Optionally, R is adapted: expanded on improvement, contracted otherwise.

    This prevents the optimizer from wandering into structurally invalid
    regions of latent space far from the wild-type, which is critical when
    the VAE latent space lacks strong KL regularization (kl_weight~0).

    Args:
        model:    Trained VAE model (BaseVAE subclass, in eval mode)
        wt_seqs:  Wild-type sequence(s) as starting point
        config:   TESS hyperparameters
        device:   Torch device
        verbose:  Print progress

    Returns:
        sequences:    Decoded protein sequences (sorted by fitness, best first)
        fitnesses:    Predicted fitness scores
        best_latents: Latent vectors of returned sequences [N, D]
    """
    model.eval()
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    best_overall_seqs = []
    best_overall_fits = []
    best_overall_z = None

    for restart in range(config.num_restarts):
        if verbose and config.num_restarts > 1:
            print(f"  [TESS] Restart {restart+1}/{config.num_restarts}")

        with torch.no_grad():
            # Step 1: Encode WT -> anchor latent
            _, mu_wt, _ = model.encode(wt_seqs)
            z_wt = mu_wt.mean(dim=0, keepdim=True)           # [1, D] anchor

        sigma = config.sigma
        trust_radius = config.trust_radius
        elites = z_wt.expand(config.top_k, -1).clone()       # [top_k, D]

        gen_best_fit = -float("inf")

        for gen in range(config.num_generations):
            with torch.no_grad():
                # Step 2a: Perturb elites
                repeats = max(1, config.population_size // config.top_k)
                z_pop = elites.repeat_interleave(repeats, dim=0)
                noise = torch.randn_like(z_pop) * sigma
                z_pop = z_pop + noise

                # TESS-specific: project back into trust region
                z_pop = _project_to_trust_region(z_pop, z_wt, trust_radius)

                # Step 2b: Predict fitness
                fitness = model.predict(z_pop)

                # Step 2c: Select top-k elites
                elites, elite_fits = _select_elites(z_pop, fitness, config.top_k)

                gen_best = elite_fits[0].item()

                # Adaptive trust radius
                if config.adaptive_trust:
                    if gen_best > gen_best_fit:
                        trust_radius = min(config.trust_max,
                                          trust_radius * config.trust_expand)
                    else:
                        trust_radius = max(config.trust_min,
                                          trust_radius * config.trust_contract)

                if gen_best > gen_best_fit:
                    gen_best_fit = gen_best

            sigma = max(config.sigma_min, sigma * config.sigma_decay)

            if verbose and (gen + 1) % 10 == 0:
                print(f"  [TESS] Gen {gen+1:3d}/{config.num_generations} | "
                      f"best_fit={gen_best_fit:.4f} | sigma={sigma:.4f} | "
                      f"trust_R={trust_radius:.3f}")

        # Decode elites
        with torch.no_grad():
            seqs = model.generate_from_latent(elites)
            fits = model.predict(elites).squeeze(-1).cpu().tolist()

        sorted_pairs = sorted(zip(seqs, fits, list(range(len(seqs)))),
                               key=lambda x: -x[1])
        seqs_sorted = [p[0] for p in sorted_pairs]
        fits_sorted = [p[1] for p in sorted_pairs]
        idx_sorted  = [p[2] for p in sorted_pairs]
        z_sorted    = elites[idx_sorted]

        if not best_overall_seqs or fits_sorted[0] > best_overall_fits[0]:
            best_overall_seqs = seqs_sorted
            best_overall_fits = fits_sorted
            best_overall_z    = z_sorted

    return best_overall_seqs, best_overall_fits, best_overall_z
