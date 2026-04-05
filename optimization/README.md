# Optimization Module: Latent Space Exploration for Protein Design

This package implements two complementary algorithms for optimizing proteins
in the latent space of a ProteinAE encoder:

1. **Gradient Ascent (gradient_ascent.py)**
   - Fast, deterministic local search
   - Uses PyTorch automatic differentiation
   - Good for quickly finding local optima

2. **Elliptical Slice Sampling (elliptical_slice_sampling.py)**
   - MCMC-based global exploration
   - No gradient computation needed
   - Discovers multiple high-scoring regions
   - Better at escaping local minima

Both algorithms optimize using a **TargetFunction** (target_function.py) that
wraps a pre-trained Oracle (TM predictor from train_tm_from_latent.py).

The **experiment_runner.py** orchestrates everything:
- Generates populations of random starting vectors
- Runs both algorithms in parallel
- Logs convergence trajectories
- Saves optimized latent vectors for validation

## Quick Start

1. Train the Oracle (if not already done):
   $ python regression/train_tm_from_latent.py \\
       --checkpoint /path/to/ae_checkpoint.ckpt \\
       --metadata fireprot_metadata.csv \\
       --structures-dir fireprot_structures

2. Run the optimization experiment:
   $ python optimization/experiment_runner.py \\
       --checkpoint regression/outputs/ridge_latent_tm_model.npz \\
       --target-tm 65.0 \\
       --n-vectors 100 \\
       --n-steps 50

3. Validate results with downstream pipeline:
   $ python validation/decode_and_validate.py \\
       --z-vectors optimization/outputs/z_final_ess.pt

## Module Components

- `target_function.TargetFunction`: Differentiable oracle wrapper
- `gradient_ascent.run_gradient_ascent()`: Baseline optimization
- `elliptical_slice_sampling.run_ess()`: MCMC-based exploration
- `experiment_runner.main()`: Master orchestration script
