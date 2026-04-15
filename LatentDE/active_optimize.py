"""
active_optimize.py - Active learning loop: iteratively run TESS, decode top
candidates, (optionally) evaluate them, and retrain the predictor head.

This implements the full LatentDE active optimization pipeline:

    repeat for num_rounds:
        1. Run TESS in latent space
        2. Decode top-k candidates -> sequences
        3. [Optional] Score sequences with an oracle (ground truth or wet lab)
        4. Add scored candidates to the training pool
        5. Retrain the predictor head (fine-tune, encoder frozen)
        6. Log results

Usage:
    python active_optimize.py --ckpt exps/ckpts/avGFP/<ckpt>.ckpt \
                               --wt_csv data/avGFP_wt.csv \
                               --num_rounds 5 \
                               --candidates_per_round 32 \
                               --oracle_csv data/avGFP_oracle.csv
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.vae_module import CNNVAE, GruVAE
from ess_tess import TESSConfig, run_tess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Active optimization loop with TESS + predictor fine-tuning"
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["cnn", "rnn"], default="cnn")
    parser.add_argument("--wt_csv", type=str, required=True,
                        help="CSV with initial WT sequences (col: sequence)")
    parser.add_argument("--oracle_csv", type=str, default=None,
                        help="CSV with (sequence, fitness) for oracle lookup. "
                             "If None, uses the model's own predictor as oracle.")
    parser.add_argument("--output_dir", type=str, default="exps/active_results")
    parser.add_argument("--device", type=str, default="cuda")

    # Active loop
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--candidates_per_round", type=int, default=32)
    parser.add_argument("--finetune_epochs", type=int, default=10,
                        help="Predictor fine-tuning epochs per round (0 = skip)")
    parser.add_argument("--finetune_lr", type=float, default=1e-4)

    # TESS config
    parser.add_argument("--population_size", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--trust_radius", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_model(ckpt_path, model_type, device):
    ModelClass = CNNVAE if model_type == "cnn" else GruVAE
    model = ModelClass.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def oracle_lookup(seqs: List[str], oracle_df: Optional[pd.DataFrame],
                  model, device: str) -> List[float]:
    """
    Return fitness scores for sequences.
    If oracle_df provided, look up ground-truth fitness.
    Otherwise, use the model predictor as a proxy oracle.
    """
    if oracle_df is not None:
        seq2fit = dict(zip(oracle_df["sequence"], oracle_df["fitness"]))
        scores = []
        for s in seqs:
            if s in seq2fit:
                scores.append(float(seq2fit[s]))
            else:
                with torch.no_grad():
                    _, mu, _ = model.encode([s])
                    pred = model.predict(mu).squeeze().item()
                scores.append(pred)
        return scores
    else:
        with torch.no_grad():
            _, mu, _ = model.encode(seqs)
            preds = model.predict(mu).squeeze(-1).cpu().tolist()
        return preds if isinstance(preds, list) else [preds]


def finetune_predictor(model, seqs: List[str], fitnesses: List[float],
                       epochs: int, lr: float, device: str):
    """Fine-tune only the predictor head on new labeled data."""
    model.freeze_encoder()
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    y = torch.tensor(fitnesses, dtype=torch.float32, device=device).unsqueeze(1)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        with torch.no_grad():
            _, mu, _ = model.encode(seqs)
        preds = model.predict(mu)
        loss = mse(preds, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    [Finetune] Epoch {epoch+1}/{epochs} | loss={loss.item():.4f}")

    model.eval()
    for param in model.encoder.parameters():
        param.requires_grad = True


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.ckpt, args.model_type, device)

    oracle_df = None
    if args.oracle_csv is not None:
        oracle_df = pd.read_csv(args.oracle_csv)
        print(f"Oracle loaded: {len(oracle_df)} entries")

    wt_df = pd.read_csv(args.wt_csv)
    wt_seqs = wt_df["sequence"].dropna().tolist()
    print(f"WT seed: {wt_seqs[0][:40]}...")

    print(f"Starting active optimization: {args.num_rounds} rounds, "
          f"{args.candidates_per_round} candidates/round")

    tess_config = TESSConfig(
        population_size=args.population_size,
        num_generations=args.num_generations,
        sigma=args.sigma,
        top_k=args.top_k,
        trust_radius=args.trust_radius,
        seed=args.seed,
    )

    all_results = []
    current_wt = wt_seqs

    for round_idx in range(args.num_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_idx+1}/{args.num_rounds}")
        print(f"{'='*60}")

        # Step 1: Run TESS
        seqs, pred_fits, _ = run_tess(
            model, current_wt, tess_config, device=device, verbose=True
        )

        # Step 2: Take top candidates
        top_seqs  = seqs[:args.candidates_per_round]
        top_preds = pred_fits[:args.candidates_per_round]

        # Step 3: Oracle scoring
        true_fits = oracle_lookup(top_seqs, oracle_df, model, device)

        # Log round results
        round_df = pd.DataFrame({
            "round": round_idx + 1,
            "sequence": top_seqs,
            "predicted_fitness": top_preds,
            "oracle_fitness": true_fits,
        })
        round_path = os.path.join(args.output_dir, f"round_{round_idx+1:02d}.csv")
        round_df.to_csv(round_path, index=False)
        all_results.append(round_df)

        best_oracle = max(true_fits)
        best_seq    = top_seqs[np.argmax(true_fits)]
        print(f"  Round {round_idx+1} best oracle fitness: {best_oracle:.4f}")

        # Steps 4+5: Fine-tune predictor
        if args.finetune_epochs > 0:
            print(f"  Fine-tuning predictor on {len(top_seqs)} new sequences...")
            finetune_predictor(
                model, top_seqs, true_fits,
                args.finetune_epochs, args.finetune_lr, device
            )

        # Step 6: Update WT seed with best found sequence
        current_wt = [best_seq]
        print(f"  New WT seed: {best_seq[:40]}...")

    # Aggregate all results
    all_df = pd.concat(all_results, ignore_index=True)
    summary_path = os.path.join(args.output_dir, "all_rounds.csv")
    all_df.to_csv(summary_path, index=False)
    print(f"\nAll results saved to {summary_path}")

    best_row = all_df.loc[all_df["oracle_fitness"].idxmax()]
    print(f"\nBest sequence found:")
    print(f"  Round: {best_row['round']}")
    print(f"  Sequence: {best_row['sequence'][:50]}...")
    print(f"  Oracle fitness: {best_row['oracle_fitness']:.4f}")
    print(f"  Predicted fitness: {best_row['predicted_fitness']:.4f}")


if __name__ == "__main__":
    main()
