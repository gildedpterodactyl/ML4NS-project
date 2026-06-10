# ProtSteer — ML for De Novo Protein Sequence Generation via MCMC Sampling

> **Steering protein sequence design using advanced Markov Chain Monte Carlo algorithms grounded in Elliptical Slice Sampling, Truncated ESS, and Transport ESS.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Background & Motivation](#background--motivation)
- [Elliptical Slice Sampling (ESS)](#elliptical-slice-sampling-ess)
- [Truncated Elliptical Slice Sampling](#truncated-elliptical-slice-sampling)
- [Transport Elliptical Slice Sampling (TESS)](#transport-elliptical-slice-sampling-tess)
- [Protein Sequence Design with PRO-LDM](#protein-sequence-design-with-pro-ldm)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [References](#references)

---

## Project Overview

**ProtSteer** investigates how modern, gradient-free MCMC methods — specifically the family of *Elliptical Slice Samplers* — can be used to steer and evaluate generative models of protein sequences. The core idea is to use sampling algorithms that can efficiently explore complex, non-Gaussian posterior distributions (as found in protein fitness landscapes and latent diffusion models) without requiring expensive gradient computations.

The project integrates three interconnected threads:

| Thread | Key Method | Goal |
|--------|-----------|------|
| Bayesian inference | Elliptical Slice Sampling (ESS) | Parameter-free posterior sampling with Gaussian priors |
| Constrained sampling | Truncated ESS | Sampling from posteriors with hard constraints / truncated supports |
| Transport-accelerated MCMC | Transport ESS (TESS) | Normalizing-flow-preconditioned ESS for complex, non-Gaussian targets |
| Protein generative modelling | PRO-LDM | Conditional latent diffusion model for protein sequence design |

---

## Background & Motivation

Probabilistic models of protein sequences must contend with an extraordinarily large and rugged fitness landscape. Standard MCMC approaches either require hand-tuning of step sizes (Metropolis–Hastings) or expensive gradient evaluations (HMC, NUTS). This is especially problematic when:

- Gradients of the likelihood are noisy or unavailable.
- The posterior has strongly non-Gaussian, funnel-like geometry.
- Parallel computation across many chains is desired (GPU/TPU settings).

Elliptical Slice Sampling and its variants address these challenges head-on with **zero tuning parameters** and **no gradient requirements**, making them natural candidates for integration with latent protein design frameworks such as PRO-LDM.

---

## Elliptical Slice Sampling (ESS)

**Paper:** Murray, Adams & MacKay, *AISTATS 2010*.

Elliptical Slice Sampling is a Markov chain Monte Carlo algorithm designed for models with **multivariate Gaussian priors**. Given a current state $f$ and prior $\mathcal{N}(0, \Sigma)$, the algorithm:

1. Draws an auxiliary variate $\nu \sim \mathcal{N}(0, \Sigma)$ to define an **ellipse** of candidate states:

$$f' = f \cos\theta + \nu \sin\theta, \quad \theta \in [0, 2\pi)$$

2. Sets a **likelihood threshold** via slice sampling: draws $u \sim \text{Uniform}(0,1)$ and requires $\log L(f') > \log L(f) + \log u$.

3. Performs a **shrinking bracket search** over $\theta$ until an acceptable proposal is found — no rejections of the final accepted state occur.

### Key Properties

- **No free parameters.** Unlike Metropolis–Hastings, there is no step-size to tune.
- **Always accepts.** The bracket shrinks until a valid sample is found; the final state is never the same as the initial state unless all points on the ellipse have zero likelihood.
- **Exact.** The algorithm is reversible and leaves the target posterior invariant.
- **Efficient in high dimensions.** The ellipse parameterisation is particularly effective when the prior is informative, as is common in Gaussian process models.

### Algorithm (Figure 2, Murray et al. 2010)

```
Input: current state f, Gaussian sampler N(0, Σ), log-likelihood log L

1.  ν ~ N(0, Σ)                        # draw auxiliary variate (defines the ellipse)
2.  u ~ Uniform(0,1)
    log y = log L(f) + log u            # slice threshold
3.  θ ~ Uniform(0, 2π)
    [θ_min, θ_max] = [θ − 2π, θ]       # initial bracket
4.  f' = f cos θ + ν sin θ
5.  if log L(f') > log y:
        return f'
    else:
        shrink bracket at θ, sample new θ from [θ_min, θ_max]
        go to 4
```

### Empirical Performance

ESS outperforms standard Metropolis–Hastings and naive Gibbs sampling on several Gaussian process models (regression, classification, log-Gaussian Cox processes), achieving more effective samples while requiring no tuning and being robust to changing hyperparameters.

---

## Truncated Elliptical Slice Sampling

Truncated ESS generalises the standard algorithm to posteriors with **hard constraints or truncated supports** — situations that commonly arise in Bayesian models with inequality constraints, simplex-valued parameters, or bounded latent variables.

The key modification is that the ellipse search must be restricted to the feasible region defined by the constraint. This is achieved by:

- Tracking the **feasible arc** of the ellipse (the subset of $\theta$ values for which the proposed $f'$ lies within the support).
- Restricting the bracket $[\theta_{\min}, \theta_{\max}]$ to the feasible arc before applying slice sampling.

This ensures that both the likelihood threshold and the hard constraint are satisfied simultaneously, while preserving the zero-tuning-parameter property of standard ESS. The truncated variant is particularly relevant for:

- Sampling **simplex-constrained** distributions (e.g., Dirichlet posteriors).
- Models with **non-negativity constraints** on latent forces or intensities.
- Protein sequence models with **conservation constraints** at specific positions.

---

## Transport Elliptical Slice Sampling (TESS)

**Paper:** Cabezas & Nemeth, *AISTATS 2023*. [[arXiv:2210.10644]](https://arxiv.org/abs/2210.10644)

TESS extends ESS to **arbitrary, non-Gaussian target distributions** by combining it with **normalizing flows** (transport maps). The central idea:

> Learn a diffeomorphism $T_\phi$ that maps the non-Gaussian target $\pi(x)$ to an approximately Gaussian reference distribution $\gamma(u)$. Then run ESS in the reference (transformed) space, and map accepted samples back to the original space via $T_\phi^{-1}$.

### Algorithm

**Step 1 — Map optimisation.** Minimise the KL divergence between the pull-back target and the reference Gaussian using stochastic gradient descent on normalizing flow parameters $\phi$:

$$\phi^* = \arg\min_\phi \, \mathrm{KL}\!\left[\pi \| T_\phi^* \gamma\right] \approx \arg\min_\phi \frac{1}{k}\sum_{i=1}^k \left[\log \gamma(u_i) - \log \pi(T_\phi(u_i))\right]$$

**Step 2 — Sampling.** Run ESS on the extended state space $(u, v)$ where $u = T_\phi^{-1}(x)$ and $v \sim \mathcal{N}(0, I)$:

```
Algorithm 1 — Transport Elliptical Slice Sampler (TESS)
Input: u, transport map T_φ

1.  v ~ N(0, I_d)
2.  w ~ Uniform(0, 1)
    log s = log γ(u) + log γ(v) + log w      # slice threshold in reference space
3.  θ ~ Uniform(0, 2π)
    [θ_min, θ_max] = [θ − 2π, θ]
4.  u' = u cos θ + v sin θ
    v' = v cos θ − u sin θ
5.  if log γ(u') + log γ(v') > log s:
        x = T_φ(u')
        return x, u'
    else:
        shrink bracket at θ, go to 4
```

**Algorithm 2 — Adaptive TESS** alternates between running $k$ parallel chains for sampling and updating $\phi$ via SGD on the KL loss, warming up for $h$ epochs before fixing the map and collecting $N$ posterior samples.

### Coupling Architecture

The normalizing flow uses **affine coupling layers** (NICE/RealNVP style). For partition $(x_A, x_B)$:

$$x_A = t_\theta(u_A \mid u_B) \odot e^{s_\theta(u_B)} + u_A, \qquad x_B = u_B$$

The Jacobian determinant is inexpensive to compute: $\det = \prod_i e^{s_i}$. Multiple coupling layers are composed to achieve flexible, high-dimensional transport.

### Performance Highlights

TESS was benchmarked against MEADS, ChEES-HMC, NUTS, and NeuTra-HMC on four challenging posterior distributions:

| Model | TESS ESS/sec | Best Competitor ESS/sec | Improvement |
|-------|-------------|------------------------|-------------|
| Biochemical oxygen demand | **1129.2** | 224.3 (ChEES) | ~5× |
| Regime switching HMM | **986.0** | 320.0 (MEADS) | ~3× |
| Predator–prey ODE | **1023.5** | ~350 | ~3× |
| Sparse logistic regression | 34.7 | **81.6** (ChEES) | — |

TESS excels on models with **rapidly changing correlation structure** where gradient-based methods are mislead by local geometry. It struggles on high-dimensional models where the normalizing flow is insufficiently expressive.

### Why TESS for Protein Design?

- Protein fitness posteriors are highly non-Gaussian with complex multimodal structure — exactly the regime where TESS excels.
- TESS requires **no target gradients**, making it applicable even when the likelihood (e.g., an ODE-based or experimental fitness model) is non-differentiable.
- It supports **parallel chains on GPUs/TPUs**, matching modern protein design workflows.

---

## Protein Sequence Design with PRO-LDM

**Paper:** Zhang et al., *bioRxiv 2023*. [[doi:10.1101/2023.08.22.554145]](https://doi.org/10.1101/2023.08.22.554145)

PRO-LDM (*Protein Sequence Generation with a conditional Latent Diffusion Model*) is the generative backbone that ProtSteer targets for steering and evaluation.

### Architecture

PRO-LDM combines three components:

```
Input sequence
      │
      ▼
┌─────────────────────────────────┐
│  Jointly Trained Autoencoder    │
│  (JT-AE)                        │
│                                 │
│  Transformer encoder (4-head,   │
│  6 layers, dim=200)             │
│       │                         │
│  Bottleneck → latent z ∈ R^64   │
│       │                         │
│  CNN decoder (4 layers)         │
│  MLP regressor (fitness)        │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Latent Conditional Diffusion   │
│  (UNet backbone, DDPM-style)    │
│                                 │
│  Learns p(z | label)            │
│  Classifier-free guidance       │
│  with guidance strength ω       │
└─────────────────────────────────┘
```

### Capabilities

- **Unconditional design:** Generates sequences matching the training distribution with higher diversity than VAE baselines.
- **Conditional design:** Generates sequences towards a target fitness label by setting the condition input.
- **Out-of-distribution (OOD) design:** Increasing the guidance strength $\omega$ steers generation away from the training distribution, enabling exploration of novel sequence space with maintained foldability (up to $\omega \approx 20$).
- **Fitness prediction:** The integrated MLP regressor predicts fitness alongside generation.

### Connection to ESS/TESS

ProtSteer treats the PRO-LDM latent space as the target distribution and applies TESS to:

1. **Steer sampling** in the latent space towards high-fitness regions without gradients of the fitness oracle.
2. **Evaluate posterior uncertainty** in the fitness landscape using MCMC diagnostics (ESS, autocorrelation time, Stein discrepancy).
3. **Compare** gradient-free TESS with gradient-based latent optimisation methods (e.g., ReLSO).

---

## Repository Structure

```
ML4NS-project/
│
├── PLDM_COMPARISON_SUITE/      # Benchmarking scripts comparing PRO-LDM variants
│                               # and MCMC steering strategies
│
├── README.md                   # This file
```

> **Note:** Additional modules (ESS implementations, TESS flow training, protein dataset pipelines) are in active development. See open branches for work-in-progress code.

---

## Getting Started

### Dependencies

```bash
pip install torch jax flax numpyro blackjax normflows biopython
```

### Running TESS on a Test Posterior

```python
from tess import TransportESS, CouplingNF

# Define target log-prob (no gradient required)
def log_target(x):
    return banana_log_prob(x)

# Initialise normalizing flow and TESS
flow = CouplingNF(dim=2, n_layers=4)
sampler = TransportESS(log_target, flow, n_chains=128)

# Warm-up and sample
sampler.warmup(n_epochs=10, n_steps=400)
samples = sampler.sample(n_steps=100)
```

### Running PRO-LDM Steering

```python
from pldm import PROLDM
from tess import LatentTESS

model = PROLDM.from_pretrained("checkpoints/gfp")
steerer = LatentTESS(model.latent_log_prob, model.flow)

# Generate high-fitness sequences via TESS steering
latents = steerer.sample(label=8, n_samples=64)
sequences = model.decode(latents)
```

---

## References

1. **Murray, I., Adams, R. P., & MacKay, D. J. C.** (2010). Elliptical slice sampling. *AISTATS*, JMLR W&CP 9, 541–548.

2. **Cabezas, A., & Nemeth, C.** (2023). Transport Elliptical Slice Sampling. *AISTATS*, PMLR 206. [arXiv:2210.10644](https://arxiv.org/abs/2210.10644)

3. **Zhang, S., Jiang, Z., Huang, R., et al.** (2024). PRO-LDM: Protein Sequence Generation with a Conditional Latent Diffusion Model. *bioRxiv*. [doi:10.1101/2023.08.22.554145](https://doi.org/10.1101/2023.08.22.554145)

4. **Natarovskii, V., Rudolf, D., & Sprungk, B.** (2021). Geometric convergence of elliptical slice sampling. *ICML*, PMLR.

5. **Hoffman, M. D., & Sountsov, P.** (2022). Tuning-free generalised Hamiltonian Monte Carlo. *AISTATS*, PMLR.

6. **Hoffman, M., Radul, A., & Sountsov, P.** (2021). An adaptive-MCMC scheme for setting trajectory lengths in Hamiltonian Monte Carlo. *AISTATS*, PMLR.

7. **Dinh, L., Krueger, D., & Bengio, Y.** (2014). NICE: Non-linear independent components estimation. *arXiv:1410.8516*.

---

*ProtSteer is a research project. Contributions, issues, and discussions are welcome.*
