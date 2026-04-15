"""
optimize.py - Run ESS or TESS latent-space optimization on a trained LatentDE model.

Usage:
    python optimize.py --method ess --ckpt exps/ckpts/avGFP/<ckpt>.ckpt \
                       --dataset avGFP --output exps/results/avGFP_ess.csv

    python optimize.py --method tess --ckpt exps/ckpts/avGFP/<ckpt>.ckpt \
                       --dataset avGFP --output exps/results/avGFP_tess.csv \
                       --trust_radius 4.0
"""

import os
import sys
import argparse
import csv
import torch
import pandas as pd
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.vae_module import CNNVAE, GruVAE
from ess_tess import ESSConfig, TESSConfig, run_ess, run_tess


def parse_args():
    parser = argparse.ArgumentParser(description="Latent-space protein optimization via ESS/TESS")

    # Core
    parser.add_argument("--method", type=str, choices=["ess", "tess"], default="tess")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained VAE checkpoint (.ckpt)")
    parser.add_argument("--model_type", type=str, choices=["cnn", "rnn"], default="cnn")
    parser.add_argument("--dataset", type=str, default="avGFP")
    parser.add_argument("--wt_csv", type=str, default=None,
                        help="CSV with wild-type sequences (col: sequence).")
    parser.add_argument("--output", type=str, default="exps/results/optimized.csv")
    parser.add_argument("--device", type=str, default="cuda")

    # ESS/TESS hyperparameters
    parser.add_argument("--population_size", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--sigma_decay", type=float, default=0.99)
    parser.add_argument("--sigma_min", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--num_restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # TESS-specific
    parser.add_argument("--trust_radius", type=float, default=4.0)
    parser.add_argument("--adaptive_trust", action="store_true", default=True)
    parser.add_argument("--trust_min", type=float, default=1.0)
    parser.add_argument("--trust_max", type=float, default=10.0)

    return parser.parse_args()


def load_model(ckpt_path: str, model_type: str, device: str):
    """Load CNNVAE or GruVAE from Lightning checkpoint."""
    ModelClass = CNNVAE if model_type == "cnn" else GruVAE
    model = ModelClass.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    print(f"Loaded {model_type.upper()}VAE from {ckpt_path}")
    return model


def get_wt_sequences(args) -> List[str]:
    """Load wild-type sequences from CSV."""
    if args.wt_csv is not None:
        df = pd.read_csv(args.wt_csv)
        assert "sequence" in df.columns, "WT CSV must have a 'sequence' column"
        seqs = df["sequence"].dropna().tolist()
        print(f"Loaded {len(seqs)} WT sequences from {args.wt_csv}")
        return seqs
    raise RuntimeError("Provide --wt_csv with a CSV containing a 'sequence' column.")


def save_results(seqs: List[str], fitnesses: List[float], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "sequence", "predicted_fitness"])
        for i, (s, fit) in enumerate(zip(seqs, fitnesses)):
            writer.writerow([i + 1, s, f"{fit:.6f}"])
    print(f"Saved {len(seqs)} optimized sequences to {output_path}")


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    model = load_model(args.ckpt, args.model_type, device)
    wt_seqs = get_wt_sequences(args)

    if args.method == "ess":
        config = ESSConfig(
            population_size=args.population_size,
            num_generations=args.num_generations,
            sigma=args.sigma,
            sigma_decay=args.sigma_decay,
            sigma_min=args.sigma_min,
            top_k=args.top_k,
            num_restarts=args.num_restarts,
            seed=args.seed,
        )
        print(f"Running ESS: pop={config.population_size}, "
              f"gens={config.num_generations}, sigma={config.sigma}")
        seqs, fits, _ = run_ess(model, wt_seqs, config, device=device, verbose=True)

    else:  # tess
        config = TESSConfig(
            population_size=args.population_size,
            num_generations=args.num_generations,
            sigma=args.sigma,
            sigma_decay=args.sigma_decay,
            sigma_min=args.sigma_min,
            top_k=args.top_k,
            num_restarts=args.num_restarts,
            seed=args.seed,
            trust_radius=args.trust_radius,
            adaptive_trust=args.adaptive_trust,
            trust_min=args.trust_min,
            trust_max=args.trust_max,
        )
        print(f"Running TESS: pop={config.population_size}, "
              f"gens={config.num_generations}, sigma={config.sigma}, "
              f"trust_R={config.trust_radius}")
        seqs, fits, _ = run_tess(model, wt_seqs, config, device=device, verbose=True)

    save_results(seqs, fits, args.output)


if __name__ == "__main__":
    main()
