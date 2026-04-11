#!/usr/bin/env python3
"""
Train a simple regression model on ProteinAE latent embeddings to predict a scalar property.

Pipeline:
1) Load ProteinAE checkpoint
2) Run encoder on structures listed in a CSV or JSON metadata file
3) Pool residue-level latent embeddings to fixed-length vectors
4) Train a ridge linear regressor (closed-form)
5) Report metrics and save artifacts

Example:
    python regression/train_tm_from_latent.py \
        --checkpoint /path/to/model.ckpt \
        --metadata /home/rohitj/Documents/courses/MLNS/fireprot_metadata.csv \
        --target-column tm \
        --structures-dir /home/rohitj/Documents/courses/MLNS/fireprot_structures
"""

from __future__ import annotations

import argparse
import gc
import json
import re
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
        description="Train a ridge regressor from ProteinAE latent representations."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to ProteinAE .ckpt")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("fireprot_metadata.csv"),
        help="CSV or JSON metadata with paths and target values",
    )
    parser.add_argument(
        "--metadata-format",
        type=str,
        default="auto",
        choices=["auto", "csv", "json"],
        help="Metadata format. Auto-detects from the file extension when possible.",
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=Path("fireprot_structures"),
        help="Directory containing structure files referenced by CSV metadata",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="radius_of_gyration",
        help="Target property column/key to predict",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="ID column for CSV metadata",
    )
    parser.add_argument(
        "--filename-column",
        type=str,
        default="filename",
        help="Filename column for CSV metadata",
    )
    parser.add_argument(
        "--path-column",
        type=str,
        default="local_path",
        help="Optional path column/key used when present",
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
        "--output-name",
        type=str,
        default=None,
        help="Optional explicit filename for the saved ridge model (.npz)",
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
    parser.add_argument(
        "--max-structure-bytes",
        type=int,
        default=0,
        help="Skip structures larger than this many bytes (0 disables size filtering)",
    )
    parser.add_argument(
        "--cpu-fallback-on-oom",
        type=int,
        choices=[0, 1],
        default=1,
        help="When 1, retry CUDA OOM samples on CPU; when 0, skip them immediately",
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
    target: float
    path: Path


def load_metadata(
    metadata_path: Path,
    structures_dir: Path,
    *,
    metadata_format: str,
    id_column: str,
    filename_column: str,
    target_column: str,
    path_column: str,
) -> List[DatasetItem]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    resolved_format = metadata_format
    if resolved_format == "auto":
        resolved_format = "json" if metadata_path.suffix.lower() == ".json" else "csv"

    items: List[DatasetItem] = []

    if resolved_format == "json":
        with metadata_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError("JSON metadata must be an object mapping ids to sample records")

        for protein_id, record in raw.items():
            if not isinstance(record, dict):
                continue
            if target_column not in record:
                continue

            path_value = record.get(path_column)
            filename_value = record.get(filename_column, protein_id)
            path = Path(path_value) if path_value else structures_dir / str(filename_value)
            if not path.exists():
                continue

            items.append(
                DatasetItem(
                    protein_id=str(protein_id),
                    filename=str(filename_value),
                    target=float(record[target_column]),
                    path=path,
                )
            )

    elif resolved_format == "csv":
        if not structures_dir.exists():
            raise FileNotFoundError(f"Structures directory not found: {structures_dir}")

        df = pd.read_csv(metadata_path)
        required_cols = {id_column, filename_column, target_column}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

        for _, row in df.iterrows():
            filename = str(row[filename_column])
            path = Path(row[path_column]) if path_column in df.columns and pd.notna(row.get(path_column)) else structures_dir / filename
            if not path.exists():
                continue
            items.append(
                DatasetItem(
                    protein_id=str(row[id_column]),
                    filename=filename,
                    target=float(row[target_column]),
                    path=path,
                )
            )
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_format}")

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


def is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def parse_size_to_gib(value: str, unit: str) -> float:
    v = float(value)
    u = unit.lower()
    if u == "gib":
        return v
    if u == "mib":
        return v / 1024.0
    return v


def is_impossible_vram_oom(exc: Exception) -> bool:
    if not is_cuda_oom(exc):
        return False

    msg = str(exc)
    tried_match = re.search(r"Tried to allocate\s+([0-9]+(?:\.[0-9]+)?)\s*(GiB|MiB)", msg)
    total_match = re.search(r"total capacity of\s+([0-9]+(?:\.[0-9]+)?)\s*(GiB|MiB)", msg)
    if not tried_match or not total_match:
        return False

    tried_gib = parse_size_to_gib(tried_match.group(1), tried_match.group(2))
    total_gib = parse_size_to_gib(total_match.group(1), total_match.group(2))
    return tried_gib > total_gib


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
    cpu_autoencoder = None

    samples = load_metadata(
        args.metadata,
        args.structures_dir,
        metadata_format=args.metadata_format,
        id_column=args.id_column,
        filename_column=args.filename_column,
        target_column=args.target_column,
        path_column=args.path_column,
    )
    print(f"Matched samples: {len(samples)}")

    feature_rows: List[np.ndarray] = []
    target_values: List[float] = []
    ids: List[str] = []
    files: List[str] = []

    oom_retry_success_count = 0
    oom_fallback_count = 0
    skipped_oom_impossible_count = 0
    skipped_oom_count = 0
    skipped_size_count = 0
    skipped_count = 0

    with torch.no_grad():
        for i, item in enumerate(samples, start=1):
            if args.max_structure_bytes > 0:
                try:
                    size_bytes = item.path.stat().st_size
                except OSError as exc:
                    skipped_count += 1
                    print(f"[WARN] Skipping {item.filename}: failed to stat file ({exc})")
                    if i % 10 == 0 or i == len(samples):
                        print(f"Encoded {i}/{len(samples)} proteins")
                    continue

                if size_bytes > args.max_structure_bytes:
                    skipped_size_count += 1
                    print(
                        f"[WARN] Skipping {item.filename}: structure size {size_bytes} bytes exceeds "
                        f"limit {args.max_structure_bytes}"
                    )
                    if i % 10 == 0 or i == len(samples):
                        print(f"Encoded {i}/{len(samples)} proteins")
                    continue

            try:
                latent = autoencoder.encode(item.path)
                feat = pool_latent(latent, pooling=args.pooling)
                feature_rows.append(feat)
                target_values.append(item.target)
                ids.append(item.protein_id)
                files.append(item.filename)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                if device.type == "cuda" and is_cuda_oom(exc):
                    print(f"[WARN] CUDA OOM on {item.filename}; clearing cache and retrying on GPU...")
                    try:
                        gc.collect()
                        torch.cuda.empty_cache()
                        latent = autoencoder.encode(item.path)
                        feat = pool_latent(latent, pooling=args.pooling)
                        feature_rows.append(feat)
                        target_values.append(item.target)
                        ids.append(item.protein_id)
                        files.append(item.filename)
                        oom_retry_success_count += 1
                        continue
                    except Exception as retry_exc:  # pragma: no cover - defensive runtime guard
                        if not is_cuda_oom(retry_exc):
                            skipped_count += 1
                            print(f"[WARN] Skipping {item.filename} after retry failed: {retry_exc}")
                            continue

                        if is_impossible_vram_oom(retry_exc) or is_impossible_vram_oom(exc):
                            skipped_oom_impossible_count += 1
                            print(
                                f"[WARN] Skipping {item.filename}: OOM request exceeds total GPU VRAM "
                                f"(cannot fit on this device)"
                            )
                            continue

                    if args.cpu_fallback_on_oom != 1:
                        skipped_oom_count += 1
                        print(f"[WARN] Skipping {item.filename}: CUDA OOM after retry (CPU fallback disabled)")
                        continue

                    print(f"[WARN] GPU retry still OOM for {item.filename}; retrying on CPU...")
                    try:
                        if cpu_autoencoder is None:
                            cpu_model = ProteinAE.load_from_checkpoint(
                                str(args.checkpoint),
                                strict=True,
                                map_location="cpu",
                            )
                            cpu_model = cpu_model.to("cpu")
                            cpu_model.eval()
                            cpu_autoencoder = ProteinAutoEncoder(model=cpu_model, trainer=None)

                        gc.collect()
                        torch.cuda.empty_cache()
                        latent = cpu_autoencoder.encode(item.path)
                        feat = pool_latent(latent, pooling=args.pooling)
                        feature_rows.append(feat)
                        target_values.append(item.target)
                        ids.append(item.protein_id)
                        files.append(item.filename)
                        oom_fallback_count += 1
                    except Exception as cpu_exc:  # pragma: no cover - defensive runtime guard
                        skipped_count += 1
                        print(f"[WARN] Skipping {item.filename} after CPU fallback failed: {cpu_exc}")
                else:
                    skipped_count += 1
                    print(f"[WARN] Skipping {item.filename}: {exc}")

            if i % 10 == 0 or i == len(samples):
                print(f"Encoded {i}/{len(samples)} proteins")

    if oom_retry_success_count > 0:
        print(f"Recovered {oom_retry_success_count} proteins by clearing CUDA cache and retrying on GPU")
    if oom_fallback_count > 0:
        print(f"Recovered {oom_fallback_count} proteins via CPU fallback after CUDA OOM")
    if skipped_oom_impossible_count > 0:
        print(f"Skipped impossible-VRAM proteins: {skipped_oom_impossible_count}")
    if skipped_oom_count > 0:
        print(f"Skipped CUDA OOM proteins (no CPU fallback): {skipped_oom_count}")
    if skipped_size_count > 0:
        print(f"Skipped oversized proteins: {skipped_size_count}")
    if skipped_count > 0:
        print(f"Skipped proteins: {skipped_count}")

    if len(feature_rows) < 3:
        raise RuntimeError("Not enough encodable proteins to train regression")

    X = np.stack(feature_rows, axis=0)
    y = np.asarray(target_values, dtype=np.float64)

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

    model_out = args.output_dir / (args.output_name or f"ridge_latent_{args.target_column}_model.npz")
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
        target_column=np.array([args.target_column]),
    )

    preds_df = pd.DataFrame(
        {
            "id": np.asarray(ids)[test_idx],
            "filename": np.asarray(files)[test_idx],
            f"{args.target_column}_true": y_test,
            f"{args.target_column}_pred": y_test_pred,
            "split": "test",
        }
    )
    preds_csv = args.output_dir / "test_predictions.csv"
    preds_df.to_csv(preds_csv, index=False)

    report = {
        "n_samples_total": int(len(y)),
        "n_oom_gpu_retry_success": int(oom_retry_success_count),
        "n_oom_cpu_fallback": int(oom_fallback_count),
        "n_skipped_oom_impossible": int(skipped_oom_impossible_count),
        "n_skipped_oom": int(skipped_oom_count),
        "n_skipped_oversize": int(skipped_size_count),
        "n_skipped": int(skipped_count),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "pooling": args.pooling,
        "ridge_alpha": float(args.ridge_alpha),
        "target_column": args.target_column,
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
