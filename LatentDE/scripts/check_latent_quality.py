import argparse
import os
import sys
from typing import List

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vae_module import CNNVAE, GruVAE  # noqa: E402
from src.common.constants import CANONICAL_ALPHABET  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Fail-safe latent quality checks")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["cnn", "rnn"], required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--random_samples", type=int, default=64)

    parser.add_argument("--max_abs_mean", type=float, default=1.0)
    parser.add_argument("--min_std", type=float, default=0.25)
    parser.add_argument("--max_std", type=float, default=3.0)
    parser.add_argument("--min_random_valid", type=float, default=0.6)
    parser.add_argument("--max_local_div", type=float, default=0.55)
    parser.add_argument("--eps", type=float, default=0.10)

    return parser.parse_args()


def load_model(ckpt_path: str, model_type: str, device: str):
    model_cls = CNNVAE if model_type == "cnn" else GruVAE
    model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def load_sequences(csv_path: str, n: int) -> List[str]:
    df = pd.read_csv(csv_path)
    seqs = df["sequence"].dropna().astype(str).tolist()
    return seqs[:n]


@torch.no_grad()
def encode_all(model, seqs: List[str], batch_size: int = 16):
    latents = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        z, *_ = model.encode(batch)
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def is_valid_protein(seq: str) -> bool:
    if not seq:
        return False
    valid = set(CANONICAL_ALPHABET)
    return all(ch in valid for ch in seq)


@torch.no_grad()
def random_validity(model, latent_dim: int, n: int, device: str) -> float:
    z = torch.randn(n, latent_dim, device=device)
    seqs = model.generate_from_latent(z)
    num_valid = sum(1 for s in seqs if is_valid_protein(s))
    return num_valid / max(1, len(seqs))


@torch.no_grad()
def local_divergence(model, seqs: List[str], eps: float, device: str, n: int = 32) -> float:
    seqs = seqs[:n]
    z = encode_all(model, seqs).to(device)

    z_pert = z + eps * torch.randn_like(z)
    s0 = model.generate_from_latent(z)
    s1 = model.generate_from_latent(z_pert)

    divs = []
    for a, b in zip(s0, s1):
        if len(a) == 0:
            continue
        m = min(len(a), len(b))
        if m == 0:
            continue
        diff = sum(1 for i in range(m) if a[i] != b[i]) / m
        divs.append(diff)
    return float(sum(divs) / max(1, len(divs)))


def main():
    args = parse_args()
    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    seqs = load_sequences(args.csv, args.num_samples)
    if len(seqs) < 8:
        print("[latent-check] FAIL: need at least 8 sequences in CSV")
        raise SystemExit(2)

    model = load_model(args.ckpt, args.model_type, device)
    latents = encode_all(model, seqs)

    mean_abs_max = latents.mean(dim=0).abs().max().item()
    std_mean = latents.std(dim=0).mean().item()
    latent_dim = latents.shape[1]

    random_valid = random_validity(model, latent_dim, args.random_samples, device)
    local_div = local_divergence(model, seqs, args.eps, device)

    print("[latent-check] metrics")
    print(f"  mean_abs_max={mean_abs_max:.4f}")
    print(f"  std_mean={std_mean:.4f}")
    print(f"  random_validity={random_valid:.4f}")
    print(f"  local_divergence@eps={args.eps:.3f}: {local_div:.4f}")

    failures = []
    if mean_abs_max > args.max_abs_mean:
        failures.append(f"mean_abs_max {mean_abs_max:.4f} > {args.max_abs_mean}")
    if std_mean < args.min_std:
        failures.append(f"std_mean {std_mean:.4f} < {args.min_std}")
    if std_mean > args.max_std:
        failures.append(f"std_mean {std_mean:.4f} > {args.max_std}")
    if random_valid < args.min_random_valid:
        failures.append(f"random_validity {random_valid:.4f} < {args.min_random_valid}")
    if local_div > args.max_local_div:
        failures.append(f"local_divergence {local_div:.4f} > {args.max_local_div}")

    if failures:
        print("[latent-check] FAIL")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(2)

    print("[latent-check] PASS")


if __name__ == "__main__":
    main()
