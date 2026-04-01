# Project Plan: Steerable Latent Diffusion for ProteinAE

**Lead R:** Vishak Kashyap K, Rohit Jeswanth
**Institution:** International Institute of Information Technology (IIIT), Hyderabad
**Focus:** High-Fidelity Protein Structure Generation via Enhanced Latent Diffusion

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Baseline Architecture Analysis](#2-baseline-architecture-analysis)
3. [Target Architecture](#3-target-architecture)
4. [Implementation Phases](#4-implementation-phases)
5. [File-Level Change Map](#5-file-level-change-map)
6. [Dataset Acquisition & Usage](#6-dataset-acquisition--usage)
7. [Compute Requirements & Time Estimates](#7-compute-requirements--time-estimates)
8. [Evaluation & Benchmarking](#8-evaluation--benchmarking)

---

## 1. Executive Summary

This project upgrades the **Latent Diffusion Model (LDM)** component of ProteinAE while keeping the pretrained autoencoder (encoder + decoder) **completely frozen**. The improvements span four phases:

| Phase | Component | Type | Compute |
|:------|:----------|:-----|:--------|
| **Phase 1** | DiT Backbone + AdaLN-Zero | Training | Cluster (8× GPU) |
| **Phase 2** | SCoT Consistency Regularizer | Training | Cluster (8× GPU) |
| **Phase 3** | CFG Rescaling + Dynamic Thresholding | Inference-only | Local (RTX 4060) |
| **Phase 4** | Training-Free Guidance (TFG) with Oracles | Inference-only | Local (RTX 4060) |

**Scope boundary:** The frozen autoencoder components (`ProteinTransformerAF3` encoder/decoder) are not modified. All changes target:
- The latent transformer backbone (`ProteinLatentTransformer`)
- The flow matching sampler (`R3NFlowMatcher`)
- The LDM training loop (`ProteinLDM.training_step`)
- The LDM inference loop (`ProteinLDM.predict_step`, `generate_latent`)
- New guidance modules (inference-only)

---

## 2. Baseline Architecture Analysis

### 2.1 Current Latent Transformer (`ProteinLatentTransformer`)

**Location:** `proteinfoundation/nn/protein_transformer.py` (lines 1270–1460)

The current LDM backbone is a 12-layer transformer operating in the latent space $z \in \mathbb{R}^{n \times 8}$:

```
Input: z_t ∈ ℝ^{n×8}  (noisy latent at time t)
  │
  ├─ linear_xt: ℝ^8 → ℝ^1152          # Project to model dimension
  ├─ + init_repr_factory(x_sc_latent)   # Self-conditioning feature
  ├─ Prepend 32 registers
  │
  ├─ 12× MultiheadAttnAndTransition     # Standard AdaLN (NOT AdaLN-Zero)
  │     ├─ AdaptiveLayerNorm(γ, β)      # Scale and shift only
  │     ├─ PairBiasAttention (disabled)  # use_attn_pair_bias: False
  │     ├─ Transition (SwiGLU)
  │     └─ Residual connections
  │
  ├─ Remove registers
  └─ linear_out: ℝ^1152 → ℝ^8          # Project back to latent dim

Output: ẑ₁ ∈ ℝ^{n×8}  (predicted clean latent)
```

**Current hyperparameters** (from `configs/experiment_config/model/ldm/pldm_r1_d8_200M.yaml`):

| Parameter | Value |
|:----------|:------|
| `token_dim` | 1152 |
| `nlayers` | 12 |
| `nheads` | 12 |
| `dim_cond` | 512 |
| `ae_token_dim` | 8 |
| `num_registers` | 32 |
| `use_attn_pair_bias` | False |
| `use_qkln` | True |
| `apply_rotary` | True |
| `parallel_mha_transition` | False (sequential) |
| Target prediction | `v` (velocity) |

### 2.2 Current Flow Matching (`R3NFlowMatcher`)

**Location:** `proteinfoundation/flow_matching/r3n_fm.py`

- **Interpolation:** Linear OT path: $z_t = (1-t) z_0 + t z_1$, where $z_0 \sim \mathcal{N}(0, I)$
- **This IS Rectified Flow** — no change needed to the interpolation scheme
- **Sampling:** 400 ODE/SDE Euler steps (`dt=0.0025`), log schedule with `p=2.0`
- **SDE mode:** Score derived from velocity via $s(z_t, t) = \frac{t \cdot v(z_t,t) - z_t}{\sigma_{\text{ref}}^2 (1-t)}$
- **Time distribution:** `mix_up02_beta(1.9, 1.0)` with shift $s=12.0$
- **Loss weighting:** $w(t) = \frac{1}{(1-t)^2 + \epsilon}$

### 2.3 Current Conditioning

- **CATH fold embedding:** 3-level (C, A, T) learned embeddings, concatenated to 768-dim, mapped to `dim_cond=512`
- **Self-conditioning:** 50% probability, feed previous prediction as `x_sc_latent`
- **CFG:** Standard double-forward — one conditioned pass, one unconditioned pass
- **Guidance formula:** $z_{\text{pred}} = w \cdot z_{\text{cond}} + (1-w)[\alpha \cdot z_{\text{ag}} + (1-\alpha) \cdot z_{\text{uncond}}]$

### 2.4 Current Training Pipeline

**Location:** `proteinfoundation/proteinflow/proteinae_ldm.py`

```
training_step(batch):
  1. Extract backbone coords → convert to nm
  2. Encode with frozen encoder → z₁ ∈ ℝ^{n×8}
  3. Sample t ~ Beta(1.9, 1.0) with shift s=12
  4. Sample z₀ ~ N(0, I)
  5. Interpolate z_t = (1-t)z₀ + t·z₁
  6. Optional: self-conditioning (50%)
  7. Predict ẑ₁ = LDM(z_t, t, cond)
  8. Loss = (1/(n·d)) Σ||z₁ - ẑ₁||² × w(t)
```

---

## 3. Target Architecture

### 3.1 DiT Backbone with AdaLN-Zero (Phase 1)

Replace the current `MultiheadAttnAndTransition` layers with **DiT blocks** using **AdaLN-Zero**:

```
Input: z_t ∈ ℝ^{n×8}
  │
  ├─ linear_xt: ℝ^8 → ℝ^1152
  ├─ + init_repr_factory(x_sc_latent)
  ├─ Prepend 32 registers
  │
  ├─ 12× DiTBlock (AdaLN-Zero):
  │     │
  │     ├─ Conditioning MLP → (γ₁, β₁, α₁, γ₂, β₂, α₂)  # 6 vectors
  │     │
  │     ├─ ATTENTION PATH:
  │     │   x' = LayerNorm(x)
  │     │   x' = γ₁ ⊙ x' + β₁                    # AdaLN modulation
  │     │   x' = MultiHeadSelfAttention(x')
  │     │   x  = x + α₁ ⊙ x'                      # Gated residual (α₁ init=0)
  │     │
  │     ├─ FFN PATH:
  │     │   x' = LayerNorm(x)
  │     │   x' = γ₂ ⊙ x' + β₂                    # AdaLN modulation
  │     │   x' = SwiGLU_FFN(x')
  │     │   x  = x + α₂ ⊙ x'                      # Gated residual (α₂ init=0)
  │     │
  │     └─ (α₁, α₂ zero-initialized → block starts as identity)
  │
  ├─ Final AdaLN + linear_out: ℝ^1152 → ℝ^8
  └─ Output: ẑ₁ ∈ ℝ^{n×8}
```

**Key mathematical difference from current AdaLN:**

| | Current (Standard AdaLN) | Target (AdaLN-Zero) |
|:--|:--|:--|
| Attention output | $x + \text{Attn}(\gamma \odot \text{LN}(x) + \beta)$ | $x + \alpha_1 \odot \text{Attn}(\gamma_1 \odot \text{LN}(x) + \beta_1)$ |
| FFN output | $x + \text{FFN}(\gamma \odot \text{LN}(x) + \beta)$ | $x + \alpha_2 \odot \text{FFN}(\gamma_2 \odot \text{LN}(x) + \beta_2)$ |
| Initialization | $\gamma=1, \beta=0$ | $\gamma=1, \beta=0, \alpha=\mathbf{0}$ |
| Effect at init | Each block is a non-trivial transform | **Each block is identity** → stable deep training |

### 3.2 SCoT — Straight-Consistent Trajectories (Phase 2)

Added as an auxiliary loss during training. For a single trajectory $(z_0, z_1)$, sample two time points $t_a, t_b$ and enforce **velocity consistency**:

$$\mathcal{L}_{\text{SCoT}} = \left\| f_\theta(z_{t_a}, t_a) - f_\theta(z_{t_b}, t_b) \right\|^2$$

where $f_\theta$ is the model's prediction of the clean sample $z_1$.

**Training objective becomes:**

$$\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda_{\text{SCoT}} \cdot \mathcal{L}_{\text{SCoT}}$$

**Effect:** After convergence, any point on a flow trajectory maps to the exact same final output. This allows **1–5 step generation** (down from 400 steps), and prevents boundary collapse near $t=1$ that degrades pLDDT scores.

### 3.3 CFG Rescaling + Dynamic Thresholding (Phase 3)

**CFG Rescaling** — in the current `predict_clean_n_v_w_guidance_latent` method:

$$z_{\text{cfg}} = z_{\text{uncond}} + w \cdot (z_{\text{cond}} - z_{\text{uncond}})$$
$$z_{\text{rescaled}} = z_{\text{cfg}} \cdot \frac{\|z_{\text{cond}}\|}{\|z_{\text{cfg}}\|}$$

**Dynamic Thresholding** — at each integration step:

$$\tau = \text{percentile}_{99.5}(|z_t|)$$
$$z_t = \text{clamp}(z_t, -\tau, \tau) / \tau$$

These prevent the severe latent space divergence observed at high guidance scales ($w > 3$).

### 3.4 Training-Free Guidance with Differentiable Oracles (Phase 4)

At each ODE integration step during inference:

$$z_{t+dt} = z_t + \left[ v_\theta(z_t, t) + \eta \cdot \nabla_{z_t} \log p(y \mid \hat{z}_1(z_t, t)) \right] \cdot dt$$

where:
- $\hat{z}_1(z_t, t) = z_t + (1-t) \cdot v_\theta(z_t, t)$ is the predicted clean latent
- $p(y \mid \hat{z}_1)$ is the oracle's property prediction
- $\eta$ is the guidance scale
- The gradient is computed w.r.t. $z_t$ only (**source-space inference** — no Jacobian through the ODE)

**Oracle integration path:**
```
z_t → v_θ(z_t, t) → ẑ₁ → Frozen Decoder → backbone coords → Oracle(coords) → ∇_{z_t} log p(y|ẑ₁)
                                                                                    ↓
                                                                          Guidance gradient
```

---

## 4. Implementation Phases

### Phase 1: DiT Backbone + AdaLN-Zero

**Objective:** Replace the latent transformer backbone with a DiT using AdaLN-Zero conditioning for stable deep training.

**Duration:** ~1 week coding + cluster training

#### Step 1.1: Create DiT Block Module

Create a new `DiTBlock` class that implements:
1. AdaLN-Zero modulation with 6 conditioning vectors per block
2. Standard multi-head self-attention (reuse existing `PairBiasAttention` or PyTorch MHA)
3. SwiGLU FFN (reuse existing `Transition`)
4. **Zero-initialized gating** parameters $\alpha_1, \alpha_2$

#### Step 1.2: Create DiT Latent Transformer

Create `DiTLatentTransformer` that:
1. Replaces the trunk `ModuleList` of `MultiheadAttnAndTransition` with `DiTBlock` layers
2. Upgrades the conditioning MLP to output 6 vectors per layer instead of a single shared conditioning vector
3. Adds a **final AdaLN** before the output projection (DiT convention)
4. Preserves the register mechanism, rotary embeddings, and self-conditioning interface

#### Step 1.3: Update Config and Integration

1. New YAML config for DiT variant
2. Update `ProteinLDM.__init__` to instantiate `DiTLatentTransformer` when config specifies DiT
3. Ensure the forward pass signature is compatible with existing `predict_clean_latent`

#### Step 1.4: Train DiT LDM from Scratch

1. Pre-encode AFDB-FS dataset to latents (one-time job)
2. Train DiT LDM on cached latents using 8 GPUs, 4h wall time with SLURM requeue
3. Validate reconstruction quality via frozen decoder

---

### Phase 2: SCoT Consistency Regularizer

**Objective:** Enforce trajectory straightness so inference reduces from 400 steps to 1–5 steps.

**Duration:** ~1 week coding + cluster fine-tuning

#### Step 2.1: Implement SCoT Loss

Create a `SCoTLoss` module that:
1. For each training sample, draws two additional time points $t_a, t_b \sim [0, 1]$
2. Computes interpolated latents $z_{t_a}, z_{t_b}$ from the same $(z_0, z_1)$ pair
3. Runs the LDM on both → obtains predicted clean samples $\hat{z}_{1|a}, \hat{z}_{1|b}$
4. Returns $\|\hat{z}_{1|a} - \hat{z}_{1|b}\|^2$

#### Step 2.2: Integrate into Training Loop

1. Add SCoT loss to `training_step` with weight $\lambda_{\text{SCoT}}$
2. Anneal $\lambda_{\text{SCoT}}$ from 0 → target value over warmup period
3. Add gradient checkpointing for the extra forward passes (memory critical)

#### Step 2.3: Update Inference for Few-Step Sampling

1. Create inference config with `dt_latent: 0.2` (5 steps) and `dt_latent: 1.0` (1 step)
2. Switch to pure ODE mode (`sampling_mode: vf`) since SCoT trajectories are straight
3. Benchmark designability vs. step count to find optimal operating point

---

### Phase 3: CFG Rescaling + Dynamic Thresholding

**Objective:** Fix structural degradation at high guidance scales for CATH-conditioned generation.

**Duration:** ~3 days coding, local testing

#### Step 3.1: CFG Rescaling

Modify `predict_clean_n_v_w_guidance_latent` in `proteinae_ldm.py`:
- After computing the guided prediction, normalize its norm to match the conditioned prediction's norm

#### Step 3.2: Dynamic Thresholding

Add a post-processing step in the sampling loop (`full_simulation` in `r3n_fm.py`):
- At each Euler step, clamp extreme latent values based on the 99.5th percentile

#### Step 3.3: Add Config Knobs

Add `cfg_rescale: True/False` and `dynamic_threshold_percentile: 0.995` to inference YAML

---

### Phase 4: Training-Free Guidance (TFG)

**Objective:** Steer generation toward functional properties (thermostability, fluorescence) without retraining.

**Duration:** ~2 weeks coding + local iteration

#### Step 4.1: Build TFG Base Framework

Create an abstract `GuidanceOracle` interface:
```python
class GuidanceOracle:
    def predict(self, backbone_coords: Tensor) -> Tensor:
        """Returns scalar property prediction (differentiable)."""
    def compute_guidance_gradient(self, z_t, t, decoder, ldm) -> Tensor:
        """Returns ∇_{z_t} log p(y | ẑ₁)."""
```

#### Step 4.2: Implement Oracle Wrappers

Wrap external predictors as differentiable PyTorch modules:
- **TemStaPro:** Load ProtT5-XL embeddings → stability classifier
- **ESMStabP:** Load ESM-2 650M → $T_m$ regressor
- **DyeLeS:** Fluorescence property predictor (if available as model weights)

#### Step 4.3: Modify Sampling Loop

In `R3NFlowMatcher.full_simulation`:
1. After each velocity prediction, optionally compute oracle guidance gradient
2. Add guidance to the velocity: $v_{\text{guided}} = v + \eta \cdot \nabla_{z_t} \log p(y | \hat{z}_1)$
3. The gradient is computed in source space (w.r.t. $z_t$) to avoid ODE Jacobian instability

#### Step 4.4: Latent Property Regression (ATLAS)

Train a lightweight regressor $f: \mathbb{R}^{n \times 8} \to \mathbb{R}^n$ that maps latent vectors directly to:
- RMSF (residue-level flexibility)
- B-factors

This avoids decoding to coordinates at every guidance step, making TFG ~20× faster.

---

## 5. File-Level Change Map

### New Files to Create

| File Path | Purpose | Phase |
|:----------|:--------|:------|
| `proteinfoundation/nn/dit.py` | `DiTBlock` and `DiTLatentTransformer` classes | Phase 1 |
| `proteinfoundation/flow_matching/scot.py` | `SCoTLoss` — trajectory consistency regularizer | Phase 2 |
| `proteinfoundation/guidance/__init__.py` | Guidance module init | Phase 4 |
| `proteinfoundation/guidance/tfg_base.py` | `GuidanceOracle` abstract base class + source-space gradient computation | Phase 4 |
| `proteinfoundation/guidance/oracles.py` | `TemStaProOracle`, `ESMStabPOracle`, `LatentPropertyOracle` wrappers | Phase 4 |
| `proteinfoundation/guidance/latent_regressor.py` | Lightweight MLP regressor $f(z) \to y$ for ATLAS RMSF prediction | Phase 4 |
| `configs/experiment_config/model/ldm/dit_pldm_200M.yaml` | DiT LDM model config | Phase 1 |
| `configs/experiment_config/training_dit_pldm_200M_afdb_512.yaml` | DiT training config | Phase 1 |
| `configs/experiment_config/training_dit_pldm_scot.yaml` | SCoT fine-tuning config | Phase 2 |
| `configs/experiment_config/inference_dit_pldm_fewstep.yaml` | Few-step (1–5) inference config | Phase 2 |
| `configs/experiment_config/inference_dit_pldm_tfg.yaml` | TFG-guided inference config | Phase 4 |
| `script_utils/pretokenize_latents.py` | Pre-encode AFDB to cached latent `.pt` files | Phase 1 (prep) |
| `script_utils/slurm_requeue.sh` | SLURM job with auto-requeue for 4h wall time | Phase 1 |
| `script_utils/train_latent_regressor.py` | Train ATLAS RMSF regressor on cached latents | Phase 4 |

### Existing Files to Modify

| File Path | Changes | Phase |
|:----------|:--------|:------|
| `proteinfoundation/nn/protein_transformer.py` | Import `DiTLatentTransformer`; keep `ProteinLatentTransformer` intact for backward compat | Phase 1 |
| `proteinfoundation/proteinflow/proteinae_ldm.py` | **`__init__`:** Conditionally instantiate `DiTLatentTransformer` vs `ProteinLatentTransformer` based on config | Phase 1 |
| `proteinfoundation/proteinflow/proteinae_ldm.py` | **`training_step` / `_compute_all_losses_latent`:** Add SCoT loss term with weight $\lambda_{\text{SCoT}}$ | Phase 2 |
| `proteinfoundation/proteinflow/proteinae_ldm.py` | **`predict_clean_n_v_w_guidance_latent`:** Add CFG rescaling (norm matching) after guidance computation | Phase 3 |
| `proteinfoundation/flow_matching/r3n_fm.py` | **`full_simulation`:** Add optional `guidance_oracle` parameter; compute + apply gradient at each step | Phase 4 |
| `proteinfoundation/flow_matching/r3n_fm.py` | **`simulation_step`:** Add dynamic thresholding option | Phase 3 |
| `proteinfoundation/nn/feature_factory.py` | **`ConditioningFactory`:** For DiT, output 6 vectors (γ₁,β₁,α₁,γ₂,β₂,α₂) per block vs current 2 (γ,β) | Phase 1 |
| `proteinfoundation/inference_ldm.py` | Add CLI args: `--guidance_oracle`, `--guidance_scale`, `--cfg_rescale`, `--dynamic_threshold` | Phase 3, 4 |
| `proteinfoundation/datasets/latent_data.py` | Support loading pre-cached latent `.pt` files for faster training | Phase 1 |
| `environment.yaml` | Add `fair-esm` (for ESMStabP oracle), `temstapro` dependencies | Phase 4 |

### Files NOT Modified (Frozen AE)

| File Path | Reason |
|:----------|:-------|
| `proteinfoundation/nn/protein_transformer.py` — `ProteinTransformerAF3` class | Frozen encoder/decoder architecture |
| `proteinfoundation/proteinflow/proteinae.py` | Autoencoder training pipeline (not used) |
| `proteinfoundation/autoencode.py` | AE inference script (unchanged) |
| `proteinfoundation/train_ae.py` | AE training script (not used) |
| `configs/experiment_config/model/ae/` | AE model configs (frozen) |
| `configs/experiment_config/training_ae_*.yaml` | AE training configs (not used) |

---

## 6. Dataset Acquisition & Usage

### 6.1 Structural Training Data

| Dataset | Purpose in Pipeline | Size | Format | Where Used | Access |
|:--------|:-------------------|:-----|:-------|:-----------|:-------|
| **AFDB-FS** (Clustered AlphaFold DB) | Primary training data for DiT LDM (Phases 1, 2) | 588,318 single-chain structures | PDB → PyG Data objects → cached latent `.pt` | `proteinfoundation/datasets/latent_data.py` — loaded during training | [AlphaFold DB](https://alphafold.ebi.ac.uk/) / [Genie2 index](https://github.com/aqlaboratory/genie2) |
| **AFDB Cluster Representatives** | Extended diversity for scaling experiments | 2.3M structures at 50% seq identity | Same pipeline | Same dataloader (optional larger training set) | [AlphaFold DB Download](https://alphafold.ebi.ac.uk/download) |

**Preprocessing for Phases 1–2:**
1. Download AFDB-FS structures (588K PDBs)
2. Filter: sequence length 32–256 residues, pLDDT > 80
3. **Pre-encode** all structures through frozen AE encoder → save as `{protein_id}_latent.pt` containing $z \in \mathbb{R}^{n \times 8}$
4. This cached dataset eliminates the encoder from the training loop, saving ~40% VRAM and ~30% wall time
5. Data augmentation: random global SO(3) rotation applied to backbone coords *before* encoding

### 6.2 Structural Classification (Conditioning)

| Dataset | Purpose in Pipeline | Size | Format | Where Used | Access |
|:--------|:-------------------|:-----|:-------|:-----------|:-------|
| **CATH Database** | Fold-conditioned generation via CATH embeddings (C, A, T levels) | ~500K domain annotations | Label mapping `.pt` file | `proteinfoundation/nn/feature_factory.py` → `FoldEmbeddingSeqFeat` → `ConditioningFactory` | [CATH DB](http://www.cathdb.info) |
| **SCOP2** | Alternative/supplementary evolutionary classification | DAG-based structural relationships | Mapping tables | Future extension of `FoldEmbeddingSeqFeat` for SCOP node conditioning | [SCOP2](http://scop2.mrc-lmb.cam.ac.uk/) |
| **SCOPe 2.07** | Berkeley extension with manual curation | Curated structural hierarchy | ASTRAL sequence libraries | Evaluation of fold novelty | [SCOPe](https://scop.berkeley.edu) |
| **SIFTS** | PDB ↔ UniProtKB residue-level mapping | All PDB entries | XML/CSV mappings | Cross-referencing oracle predictions to structure coordinates | [SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/) |

**Usage in Phases 1–2:**
- CATH labels consumed by `FoldEmbeddingSeqFeat` in `feature_factory.py`
- Fed as conditioning to DiT via the `cond_factory` → AdaLN-Zero modulation
- CFG during training: progressive masking H→T→A→C with configurable probabilities
- Loaded from `cath_label_mapping.pt` referenced in config at `cath_code_dir`

### 6.3 Structural Benchmarks (Evaluation)

| Dataset | Purpose in Pipeline | Metrics | Where Used | Access |
|:--------|:-------------------|:--------|:-----------|:-------|
| **CASP14** | Gold-standard structure reconstruction benchmark | RMSD, GDT, FAPE, pLDDT, TM-score | Evaluate AE reconstruction quality (sanity check that frozen AE still works) | [Prediction Center CASP14](https://predictioncenter.org/casp14/) |
| **CASP15** | Assemblies, ligands, ensembles evaluation | pLDDT, TM-score, Interface similarity | Evaluate generalization of generated structures | [Prediction Center CASP15](https://predictioncenter.org/casp15/) |
| **PDB** (experimental) | Ground-truth structural validation | scRMSD, TM-score | Designability metric computation | [RCSB PDB](https://www.rcsb.org/) |

**Usage in all Phases (evaluation):**
- Designability (Des): Generate 8 sequences per backbone via ProteinMPNN → fold with ESMFold → scRMSD < 2Å
- Diversity (Div): Cluster designable backbones via Foldseek (TM-score threshold 0.5)
- Novelty (Nov): Max TM-score against AFDB and PDB references
- DPT: Pairwise TM-score among designable samples
- Computed via `proteinfoundation/metrics/designability.py` and `proteinfoundation/metrics/metric_factory.py`

### 6.4 Biophysical Property Data (Oracle Training & TFG)

| Dataset | Purpose in Pipeline | Size | Property | Where Used | Access |
|:--------|:-------------------|:-----|:---------|:-----------|:-------|
| **ATLAS** (Non-Redundant Core) | Train latent-space flexibility regressor $f(z) \to \text{RMSF}$ | 1,068 proteins (core) / 1,938 (Nov 2024) | RMSF, B-factor from MD trajectories | `script_utils/train_latent_regressor.py` → produces `guidance/latent_regressor.py` weights | [ATLAS](https://www.dsimb.inserm.fr/ATLAS) |
| **TransAtlas** | Large-scale conformational transition data | 64,646 independent transitions | Conformational change magnitudes | Extended evaluation of dynamic properties | [TransAtlas](https://mmb.irbbarcelona.org/TransAtlas/) |
| **FireProtDB 2.0** | Ground truth for thermostability oracle validation | 5.5M+ experiments | $T_m$, $\Delta T_m$, $\Delta\Delta G$ | Validate TemStaPro/ESMStabP oracle accuracy before TFG deployment | [FireProtDB](https://loschmidt.chemi.muni.cz/fireprotdb/) |
| **MegaScale** | High-throughput stability measurements | ~670K mutations | Protease-derived stability proxy | Supplementary oracle validation | Included in FireProtDB 2.0 |
| **Meltome Atlas** | Species-wide melting temperatures | 48,000 proteins | $T_m$ across species | Oracle calibration | [Meltome Atlas](https://meltomeatlas.proteomics.wzw.tum.de/) |

**Usage in Phase 4 (TFG):**
1. **ATLAS → Latent Regressor:** Encode ATLAS structures with frozen AE → paired $(z, \text{RMSF})$ dataset → train lightweight MLP $f: \mathbb{R}^{n \times 8} \to \mathbb{R}^n$ → used as fast guidance oracle (avoids decoding at every step)
2. **FireProtDB → Oracle Validation:** Before deploying TemStaPro/ESMStabP as guidance oracles, validate their predictions against FireProtDB experimental $\Delta\Delta G$ values
3. **Meltome → Oracle Calibration:** Calibrate ESMStabP $T_m$ predictions against experimental melting temperatures

### 6.5 Functional Oracle Models (TFG Inference)

| Oracle Tool | Foundation Model | Output | Use in TFG | Access |
|:------------|:----------------|:-------|:-----------|:-------|
| **TemStaPro** | ProtT5-XL embeddings | Multi-threshold stability classification (40°C–80°C) | Thermostability guidance: $\nabla_{z_t} \log p(\text{stable} \mid \hat{z}_1)$ | [Zenodo](https://doi.org/10.5281/zenodo.7743637) / [GitHub](https://github.com/ievapudz/TemStaPro) |
| **ESMStabP** | ESM-2 650M (layer 33) | Predicted $T_m$ (regression) | Thermostability guidance: $\nabla_{z_t} T_m(\hat{z}_1)$ | [GitHub](https://github.com/marcusramos2024/ESMStabP) |
| **DyeLeS** | Fluorescence predictor | Stokes shift, quantum yield, classification | Fluorescence guidance: $\nabla_{z_t} \text{emission}(\hat{z}_1)$ | [DyeLeS](https://dyeles.molastra.com) / [FluorGen](https://github.com/MolAstra/FluorGen) |
| **Latent RMSF Regressor** | Trained on ATLAS (local) | Residue-level RMSF from latent | Fast flexibility guidance (no decoding needed) | Trained locally via `script_utils/train_latent_regressor.py` |

**Usage in Phase 4:**
- Wrapped as differentiable `GuidanceOracle` subclasses in `proteinfoundation/guidance/oracles.py`
- During inference, the oracle receives either:
  - **Full path** (slower, more accurate): $z_t \to \hat{z}_1 \to \text{Decoder} \to \text{coords} \to \text{Oracle} \to \nabla_{z_t}$
  - **Latent shortcut** (faster): $z_t \to \hat{z}_1 \to f(\hat{z}_1) \to \nabla_{z_t}$ (ATLAS regressor only)
- Multiple oracles can be combined: $\nabla_{z_t} = \eta_1 \nabla_{z_t} \log p_1 + \eta_2 \nabla_{z_t} \log p_2$

### 6.6 Fluorescence Design Data (Downstream Application)

| Dataset | Purpose | Size | Access |
|:--------|:--------|:-----|:-------|
| **DyeLeS / FluoBioDB** | Fluorescent molecule classification & prediction | 32,865 molecules | [DyeLeS](https://dyeles.molastra.com) |
| **FluorGen / FluoDB** | Automated dye discovery training data | 35,528 compounds | [FluorGen](https://github.com/MolAstra/FluorGen) |
| **RFP Variant Dataset** | β-barrel fluorescent chimeras, brightness + red-shift | Gene synthesis + NGS | [BioProject PRJNA1273454](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1273454) |
| **RFP Parent Libraries** | Codon-optimized gene libraries for chimeric diversity | Addgene deposits | [Addgene #245482](https://www.addgene.org/), [#245483](https://www.addgene.org/) |
| **NGS Mapping Pipeline** | Sequence-function mapping scripts for RFPs | Analysis code | [GitHub](https://github.com/PlesaLab/Fluorescent_Protein_NGS_pipeline) |

**Usage:** Downstream validation of TFG-guided designs. Generated β-barrel proteins can be scored against fluorescence property predictors trained on these datasets.

---

## 7. Compute Requirements & Time Estimates

### 7.1 Available Compute

| Resource | Specs |
|:---------|:------|
| **Cluster (CCNSB)** | 8 GPUs, 80 CPUs, 40 concurrent jobs, 4-hour wall time per job |
| **Local** | NVIDIA RTX 4060 (8GB VRAM), AMD Ryzen 7, 24GB RAM |

### 7.2 Phase-by-Phase Estimates

#### Pre-Encoding Step (One-Time)

| Item | Value |
|:-----|:------|
| **Task** | Run frozen AE encoder on 588K AFDB structures → save latent `.pt` files |
| **Compute** | Cluster, 1–2 GPUs |
| **Estimated time** | ~2–4 hours (single job, ~150 structures/sec on one GPU) |
| **Output** | ~588K `.pt` files, ~50GB total |
| **SLURM jobs** | 1 |

#### Phase 1: DiT + AdaLN-Zero Training

| Item | Value |
|:-----|:------|
| **Task** | Train DiT LDM from scratch on cached latents |
| **Compute** | Cluster, 8 GPUs (DDP) |
| **Batch size** | 8 per GPU × 8 GPUs × 2 gradient accumulation = effective batch 128 |
| **Steps to convergence** | ~500K–800K steps (estimated from baseline) |
| **Steps per 4h job** | ~25K steps (at ~0.6s/step with 8 GPUs, cached latents) |
| **SLURM jobs** | ~25–35 jobs × 4h = **100–140 GPU-hours** |
| **Wall clock** | ~4–5 days (submitting ~7 jobs/day with queue wait) |
| **Checkpoint strategy** | Save every 5K steps (`last_ckpt_every_n_steps: 5000`), SLURM `--requeue` on TIMEOUT |

#### Phase 2: SCoT Fine-Tuning

| Item | Value |
|:-----|:------|
| **Task** | Fine-tune Phase 1 checkpoint with SCoT consistency loss |
| **Compute** | Cluster, 8 GPUs (DDP) |
| **Extra cost per step** | ~2× (two additional forward passes per sample for $t_a, t_b$) |
| **Steps to convergence** | ~100K–200K steps (fine-tune, not from scratch) |
| **SLURM jobs** | ~10–15 jobs × 4h = **40–60 GPU-hours** |
| **Wall clock** | ~2–3 days |

#### Phase 3: CFG Rescaling + Dynamic Thresholding

| Item | Value |
|:-----|:------|
| **Task** | Code changes + inference testing |
| **Compute** | Local RTX 4060 |
| **Estimated time** | ~3 days development |
| **Inference per protein** | ~5–10 seconds (5-step sampling post-SCoT) |
| **VRAM usage** | ~2–3 GB (LDM inference only, no encoder needed during latent sampling) |

#### Phase 4: TFG with Oracles

| Item | Value |
|:-----|:------|
| **Task** | Implement oracle wrappers + guided inference |
| **Compute** | Local RTX 4060 |
| **Development time** | ~2 weeks |
| **ATLAS regressor training** | ~1 hour on local GPU (small MLP, 1K–2K samples) |
| **Guided inference per protein** | |
| → Latent regressor path | ~10–20 seconds (fast, no decoding) |
| → Full decoder + oracle path | ~1–3 minutes (requires decoding at each guidance step) |
| **VRAM for ESMStabP oracle** | ~2.5 GB (ESM-2 650M) + ~1 GB (LDM + decoder) = ~3.5 GB total ✅ fits 8GB |
| **VRAM for TemStaPro oracle** | ~3 GB (ProtT5-XL) + ~1 GB = ~4 GB ✅ fits 8GB |

### 7.3 Total Compute Budget Summary

| Phase | GPU-Hours | Wall Clock | Location |
|:------|:----------|:-----------|:---------|
| Pre-encoding | 4–8 | 1 day | Cluster |
| Phase 1 (DiT) | 100–140 | 4–5 days | Cluster |
| Phase 2 (SCoT) | 40–60 | 2–3 days | Cluster |
| Phase 3 (CFG) | <1 | 3 days | Local |
| Phase 4 (TFG) | ~5 (regressor) | 2 weeks | Local |
| **Total** | **~150–215 GPU-hours** | **~4–5 weeks** | |

### 7.4 SLURM Job Template

```bash
#!/bin/bash
#SBATCH --job-name=dit_pldm
#SBATCH --account=ccnsb
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --signal=SIGUSR1@300    # Signal 5 min before timeout for graceful checkpoint
#SBATCH --requeue                # Auto-requeue on timeout
#SBATCH --output=logs/dit_pldm_%j.out

# Activate environment
source activate proteinae

# Training resumes automatically from last checkpoint
python proteinfoundation/train_ldm.py \
    --config_name training_dit_pldm_200M_afdb_512
```

---

## 8. Evaluation & Benchmarking

### 8.1 Metrics

| Metric | Definition | Target | Phase |
|:-------|:-----------|:-------|:------|
| **Designability (Des)** | Fraction with scRMSD < 2Å (8 seqs/backbone via ProteinMPNN → ESMFold) | ≥ 93% (match baseline) | All |
| **Diversity (Div)** | Number of Foldseek clusters at TM-score 0.5 | ≥ 204 (match baseline) | All |
| **Novelty (Nov)** | Fraction with max TM-score < 0.5 vs AFDB+PDB | ≥ 0.70 | All |
| **DPT** | Mean pairwise TM-score among designable samples | ≤ 0.36 (lower = more diverse) | All |
| **NFE** | Number of function evaluations (ODE steps) | ≤ 5 (post-SCoT) | Phase 2 |
| **pLDDT** | AlphaFold confidence of generated structures | ≥ 80 | All |
| **Flex-RMSF $\rho$** | Spearman correlation of latent-predicted vs ATLAS RMSF | ≥ 0.65 | Phase 4 |
| **$\Delta T_m$** | TFG-guided improvement in predicted melting temperature | ≥ +20°C vs unguided | Phase 4 |

### 8.2 Baseline Comparison Table

| Model | Des ↑ | Div ↑ | DPT ↓ | Nov ↑ | NFE ↓ |
|:------|:------|:------|:------|:------|:------|
| RFdiffusion | 96% | 247 | 0.43 | 0.71 | ~100 |
| ESM3 | 61% | 127 | 0.37 | 0.84 | N/A |
| DPLM-2 (650M) | 63% | 130 | 0.37 | 0.72 | N/A |
| ProteinAE-PLDM (baseline) | 93% | 204 | 0.36 | 0.70 | 400 |
| **DiT-PLDM + SCoT (ours, target)** | **≥93%** | **≥204** | **≤0.36** | **≥0.70** | **≤5** |

### 8.3 Ablation Studies

| Ablation | Purpose |
|:---------|:--------|
| DiT vs baseline `ProteinLatentTransformer` at same FLOPs | Isolate DiT architectural benefit |
| SCoT λ sweep (0.01, 0.1, 0.5, 1.0) | Find optimal consistency weight |
| NFE sweep (1, 2, 5, 10, 20, 50, 100, 400 steps) | Characterize quality-speed tradeoff post-SCoT |
| CFG scale sweep with/without rescaling | Demonstrate CFG rescaling prevents degradation |
| TFG guidance scale $\eta$ sweep per oracle | Find Pareto front of property vs structural quality |
| Combined oracles ($\eta_{\text{stab}} + \eta_{\text{flex}}$) | Multi-objective guidance feasibility |

---

## Appendix A: Data Flow Diagram

```
                        ┌─────────────────────────────────────┐
                        │     AFDB-FS (588K structures)       │
                        │     + CATH labels                   │
                        └──────────────┬──────────────────────┘
                                       │
                            Pre-encode (one-time)
                            Frozen AE Encoder
                                       │
                                       ▼
                        ┌─────────────────────────────────────┐
                        │   Cached Latents: z ∈ ℝ^{n×8}      │
                        │   + CATH labels per structure       │
                        └──────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
         Phase 1                  Phase 2                  Phase 3-4
    Train DiT LDM           Fine-tune + SCoT          Inference Only
              │                        │                        │
              ▼                        ▼                        ▼
    ┌──────────────┐        ┌──────────────┐        ┌──────────────────┐
    │ DiT + AdaLN  │        │ + SCoT Loss  │        │ Guided Sampling  │
    │ Zero backbone│───────▶│ λ_scot * L   │───────▶│ + CFG Rescale    │
    │              │ ckpt   │              │ ckpt   │ + TFG Oracles    │
    └──────────────┘        └──────────────┘        └────────┬─────────┘
                                                             │
                                                  ┌──────────┼──────────┐
                                                  │          │          │
                                              TemStaPro  ESMStabP   ATLAS
                                              Oracle     Oracle    Regressor
                                                  │          │          │
                                                  └──────────┼──────────┘
                                                             │
                                                             ▼
                                                  ┌──────────────────┐
                                                  │  Generated z₁    │
                                                  └────────┬─────────┘
                                                           │
                                                  Frozen AE Decoder
                                                  (20 ODE steps)
                                                           │
                                                           ▼
                                                  ┌──────────────────┐
                                                  │  Backbone Coords │
                                                  │  → PDB output    │
                                                  └──────────────────┘
                                                           │
                                              ┌────────────┼────────────┐
                                              │            │            │
                                          CASP14/15     Foldseek    FireProtDB
                                          Benchmark    Clustering   Validation
                                              │            │            │
                                              ▼            ▼            ▼
                                          Des/pLDDT    Div/Nov/DPT  Oracle Acc.
```

---

## Appendix B: Complete Dataset Links

| # | Dataset / Tool | Primary Purpose | Link |
|:--|:---------------|:----------------|:-----|
| 1 | ProteinAE V1 Code | Baseline repository | https://github.com/OnlyLoveKFC/ProteinAE_v1 |
| 2 | Genie 2 Code / Index | AFDB structural data index | https://github.com/aqlaboratory/genie2 |
| 3 | AlphaFold Database | Full predicted structural prior (214M+) | https://alphafold.ebi.ac.uk/ |
| 4 | AlphaFold DB Download | Bulk download of predicted structures | https://alphafold.ebi.ac.uk/download |
| 5 | RCSB PDB | Experimental structures (200K+) | https://www.rcsb.org/ |
| 6 | CATH Database | Class, Architecture, Topology, Homology | http://www.cathdb.info |
| 7 | SCOP2 | Structural and evolutionary relationships | http://scop2.mrc-lmb.cam.ac.uk/ |
| 8 | SCOPe 2.07 | Berkeley SCOP extension | https://scop.berkeley.edu |
| 9 | ASTRAL | Sequence libraries for SCOP domains | https://scop.berkeley.edu/astral/ |
| 10 | SIFTS | PDB ↔ UniProtKB residue mapping | https://www.ebi.ac.uk/pdbe/docs/sifts/ |
| 11 | ATLAS Database | Protein flexibility (RMSF/B-factor) MD data | https://www.dsimb.inserm.fr/ATLAS |
| 12 | TransAtlas | Conformational transitions | https://mmb.irbbarcelona.org/TransAtlas/ |
| 13 | FireProtDB 2.0 | Mutational thermostability (5.5M experiments) | https://loschmidt.chemi.muni.cz/fireprotdb/ |
| 14 | Meltome Atlas | Species-wide melting temperatures | https://meltomeatlas.proteomics.wzw.tum.de/ |
| 15 | CASP14 | Structure prediction benchmark | https://predictioncenter.org/casp14/ |
| 16 | CASP15 | Structure prediction benchmark | https://predictioncenter.org/casp15/ |
| 17 | TemStaPro (Data) | Thermostability prediction training data | https://doi.org/10.5281/zenodo.7743637 |
| 18 | TemStaPro (Code) | Thermostability prediction software | https://github.com/ievapudz/TemStaPro |
| 19 | ESMStabP | ESM-2 based $T_m$ regression | https://github.com/marcusramos2024/ESMStabP |
| 20 | ESM2StabP | Random Forest on ESM-2 features | https://biolm.ai/models/esm2stabp/ |
| 21 | DyeLeS / FluoBioDB | Fluorescence property prediction (32K molecules) | https://dyeles.molastra.com |
| 22 | FluorGen / FluoDB | Fluorescent compound generation (35K) | https://github.com/MolAstra/FluorGen |
| 23 | RFP Variant Dataset | β-barrel fluorescent chimeras | NCBI BioProject PRJNA1273454 |
| 24 | RFP Parent Libraries | Codon-optimized gene libraries | Addgene #245482, #245483 |
| 25 | NGS Mapping Pipeline | Sequence-function mapping for RFPs | https://github.com/PlesaLab/Fluorescent_Protein_NGS_pipeline |
| 26 | UniRef50 | Sequence clustering at 50% identity | https://www.uniprot.org/uniref/ |
| 27 | ESM Structural Split | Superfamily/Fold training splits | https://github.com/facebookresearch/esm |

---

*Last updated: March 9, 2026*
