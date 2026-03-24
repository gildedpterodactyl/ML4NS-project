#!/usr/bin/env python3
"""
Train a simple regression model on ProteinAE latent embeddings to predict TM values.

Pipeline:
1) Load ProteinAE checkpoint
2) Run encoder on structures listed in fireprot_metadata.csv
3) Pool residue-level latent embeddings to fixed-length vectors
4) Train a ridge linear regressor (closed-form)
5) Report metrics and save artifacts

Example:
    python regression/train_tm_from_latent.py \
        --checkpoint /path/to/model.ckpt \
        --metadata /home/rohitj/Documents/courses/MLNS/fireprot_metadata.csv \
        --structures-dir /home/rohitj/Documents/courses/MLNS/fireprot_structures
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def add_project_to_path(project_root: Path) -> None:
    sys.path.insert(0, str(project_root.resolve()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a TM regressor from ProteinAE latent representations."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to ProteinAE .ckpt")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("fireprot_metadata.csv"),
        help="CSV with columns including filename and tm",
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=Path("fireprot_structures"),
        help="Directory containing structure files from metadata",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("ML4NS-project"),
        help="Path to the ML4NS project root that contains proteinfoundation package",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("regression/outputs"),
        help="Directory to save model artifacts and predictions",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Pooling method to convert residue-level latents to fixed-size vectors",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="L2 regularization coefficient for ridge regression",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for encoder inference",
    )
    return parser


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DatasetItem:
    protein_id: str
    filename: str
    tm: float
    path: Path


def load_metadata(metadata_path: Path, structures_dir: Path) -> List[DatasetItem]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if not structures_dir.exists():
        raise FileNotFoundError(f"Structures directory not found: {structures_dir}")

    df = pd.read_csv(metadata_path)
    required_cols = {"id", "filename", "tm"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    items: List[DatasetItem] = []
    for _, row in df.iterrows():
        filename = str(row["filename"])
        path = structures_dir / filename
        if not path.exists():
            continue
        items.append(
            DatasetItem(
                protein_id=str(row["id"]),
                filename=filename,
                tm=float(row["tm"]),
                path=path,
            )
        )
    if not items:
        raise RuntimeError("No valid samples found after matching metadata to structure files")
    return items


def pool_latent(single_repr: torch.Tensor, pooling: str) -> np.ndarray:
    if single_repr.ndim != 3:
        raise ValueError(f"Expected latent shape [B, L, D], got {tuple(single_repr.shape)}")

    latent = single_repr.detach().cpu()
    if pooling == "mean":
        vec = latent.mean(dim=1)
    elif pooling == "max":
        vec = latent.max(dim=1).values
    else:
        raise ValueError(f"Unsupported pooling method: {pooling}")
    return vec.squeeze(0).numpy().astype(np.float64)


def train_test_split_indices(n: int, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 3:
        raise ValueError("Need at least 3 samples to split train/test")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 1)
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    return train_idx, test_idx


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    Xs = (X - mean) / std
    return Xs, mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    n_samples, n_features = X.shape
    ones = np.ones((n_samples, 1), dtype=X.dtype)
    X_aug = np.concatenate([ones, X], axis=1)

    reg = np.eye(n_features + 1, dtype=X.dtype)
    reg[0, 0] = 0.0
    A = X_aug.T @ X_aug + alpha * reg
    b = X_aug.T @ y
    w = np.linalg.solve(A, b)

    intercept = float(w[0])
    coef = w[1:]
    return coef, intercept


def predict_linear(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return X @ coef + intercept


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    args = build_parser().parse_args()

    add_project_to_path(args.project_root)

    from proteinfoundation.autoencode import ProteinAutoEncoder  # pylint: disable=import-outside-toplevel
    from proteinfoundation.proteinflow.proteinae import ProteinAE  # pylint: disable=import-outside-toplevel

    device = pick_device(args.device)
    print(f"Using device: {device}")

    model = ProteinAE.load_from_checkpoint(
        str(args.checkpoint),
        strict=True,
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    autoencoder = ProteinAutoEncoder(model=model, trainer=None)

    samples = load_metadata(args.metadata, args.structures_dir)
    print(f"Matched samples: {len(samples)}")

    feature_rows: List[np.ndarray] = []
    tm_values: List[float] = []
    ids: List[str] = []
    files: List[str] = []

    with torch.no_grad():
        for i, item in enumerate(samples, start=1):
            try:
                latent = autoencoder.encode(item.path)
                feat = pool_latent(latent, pooling=args.pooling)
                feature_rows.append(feat)
                tm_values.append(item.tm)
                ids.append(item.protein_id)
                files.append(item.filename)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                print(f"[WARN] Skipping {item.filename}: {exc}")

            if i % 10 == 0 or i == len(samples):
                print(f"Encoded {i}/{len(samples)} proteins")

    if len(feature_rows) < 3:
        raise RuntimeError("Not enough encodable proteins to train regression")

    X = np.stack(feature_rows, axis=0)
    y = np.asarray(tm_values, dtype=np.float64)

    train_idx, test_idx = train_test_split_indices(len(y), args.test_size, args.seed)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    X_train_std, x_mean, x_std = standardize_fit(X_train)
    X_test_std = standardize_apply(X_test, x_mean, x_std)

    coef, intercept = fit_ridge_closed_form(X_train_std, y_train, alpha=args.ridge_alpha)
    y_train_pred = predict_linear(X_train_std, coef, intercept)
    y_test_pred = predict_linear(X_test_std, coef, intercept)

    train_metrics = metrics(y_train, y_train_pred)
    test_metrics = metrics(y_test, y_test_pred)

    print("\nTrain metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_out = args.output_dir / "ridge_latent_tm_model.npz"
    np.savez_compressed(
        model_out,
        coef=coef,
        intercept=np.array([intercept], dtype=np.float64),
        x_mean=x_mean,
        x_std=x_std,
        pooling=np.array([args.pooling]),
        ridge_alpha=np.array([args.ridge_alpha], dtype=np.float64),
        checkpoint=np.array([str(args.checkpoint)]),
        project_root=np.array([str(args.project_root)]),
    )

    preds_df = pd.DataFrame(
        {
            "id": np.asarray(ids)[test_idx],
            "filename": np.asarray(files)[test_idx],
            "tm_true": y_test,
            "tm_pred": y_test_pred,
            "split": "test",
        }
    )
    preds_csv = args.output_dir / "test_predictions.csv"
    preds_df.to_csv(preds_csv, index=False)

    report = {
        "n_samples_total": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "pooling": args.pooling,
        "ridge_alpha": float(args.ridge_alpha),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "metadata": str(args.metadata),
        "structures_dir": str(args.structures_dir),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "model_file": str(model_out),
        "predictions_file": str(preds_csv),
    }
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\nSaved artifacts:")
    print(f"  Model: {model_out}")
    print(f"  Predictions: {preds_csv}")
    print(f"  Metrics: {report_path}")


if __name__ == "__main__":
    main()
