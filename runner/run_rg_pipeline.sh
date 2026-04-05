#!/usr/bin/env bash
set -euo pipefail

# Radius of Gyration end-to-end runner
# Steps:
# 1) Download ~200 proteins using GPML4NS/data-collection/gyr_agg.py
# 2) Generate ~50 proteins via ProteinAE autoencode mode
# 3) Validate Rg using GPML4NS/protein-verification/gyr_pred.py
# 4) Plot and save comparisons

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPML_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$GPML_DIR/.." && pwd)"
RUNNER_DIR="$SCRIPT_DIR"
DATA_DIR="${DATA_DIR:-$RUNNER_DIR/data}"
RESULTS_DIR="${RESULTS_DIR:-$RUNNER_DIR/results}"
OUT_DIR="$RESULTS_DIR"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.11}"
UV_VENV_DIR="${UV_VENV_DIR:-$RUNNER_DIR/.venv}"
TORCH_VARIANT="${TORCH_VARIANT:-cu121}"

DOWNLOAD_DIR="$DATA_DIR/downloaded_200_pdbs"
DOWNLOAD_META="$DATA_DIR/downloaded_200_properties.json"
GENERATED_RAW_DIR="$DATA_DIR/generated_raw"
GENERATED_FLAT_DIR="$DATA_DIR/generated_50_pdbs"
VALIDATION_DIR="$RESULTS_DIR/validation"
PLOTS_DIR="$RESULTS_DIR/plots"
LOGS_DIR="$RESULTS_DIR/logs"

DOWNLOAD_N="${DOWNLOAD_N:-200}"
GENERATE_N="${GENERATE_N:-50}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$DOWNLOAD_DIR" "$GENERATED_RAW_DIR" "$GENERATED_FLAT_DIR" "$VALIDATION_DIR" "$PLOTS_DIR" "$LOGS_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed. Install uv first."
  exit 1
fi

echo "[0/6] Setting up uv environment and installing required packages..."
if [[ ! -d "$UV_VENV_DIR" ]]; then
  uv venv --python "$UV_PYTHON_VERSION" "$UV_VENV_DIR"
fi

UV_PYTHON_BIN="$UV_VENV_DIR/bin/python"

if [[ "$TORCH_VARIANT" == "cu121" ]]; then
  uv pip install --python "$UV_PYTHON_BIN" \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
elif [[ "$TORCH_VARIANT" == "cpu" ]]; then
  uv pip install --python "$UV_PYTHON_BIN" \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu
else
  echo "ERROR: TORCH_VARIANT must be one of: cu121, cpu"
  exit 1
fi

uv pip install --python "$UV_PYTHON_BIN" \
  requests numpy pandas matplotlib biopython scipy \
  lightning torch-geometric einops omegaconf hydra-core \
  loguru python-dotenv jaxtyping dm-tree cpdb-protein wget biopandas transformers timm

run_uv_py() {
  uv run --python "$UV_PYTHON_BIN" "$@"
}

echo "[1/6] Downloading about ${DOWNLOAD_N} proteins..."
(
  cd "$ROOT_DIR"
  UNIQUE_TARGET="$DOWNLOAD_N" \
  SAVE_DIR="$DOWNLOAD_DIR" \
  METADATA_FILE="$DOWNLOAD_META" \
  run_uv_py "$GPML_DIR/data-collection/gyr_agg.py"
)

echo "[2/6] Preparing model checkpoint linkage for autoencoder inference..."
MODEL_DIR="$GPML_DIR/model"
mkdir -p "$MODEL_DIR/checkpoints"

CKPT_CANDIDATES=(
  "$GPML_DIR/model-checkpoints/ae_r1_d8_v1.ckpt"
  "$ROOT_DIR/model-checkpoints/ae_r1_d8_v1.ckpt"
  "$ROOT_DIR/ae_r1_d8_v1.ckpt"
)

FOUND_CKPT=""
for p in "${CKPT_CANDIDATES[@]}"; do
  if [[ -f "$p" ]]; then
    FOUND_CKPT="$p"
    break
  fi
done

if [[ -z "$FOUND_CKPT" ]]; then
  echo "ERROR: Could not find ae_r1_d8_v1.ckpt"
  exit 1
fi

ln -sf "$FOUND_CKPT" "$MODEL_DIR/checkpoints/ae_r1_d8_v1.ckpt"
echo "Using checkpoint: $FOUND_CKPT"

echo "[2.5/6] Resolving GPU allocation for parallel generation..."
GPU_LIST_RAW="${GPU_LIST:-}"
if [[ -z "$GPU_LIST_RAW" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_LIST_RAW="$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT_DETECTED="$(nvidia-smi -L | wc -l | tr -d ' ')"
    if [[ "$GPU_COUNT_DETECTED" -gt 0 ]]; then
      GPU_LIST_RAW="$(seq -s, 0 $((GPU_COUNT_DETECTED - 1)))"
    else
      GPU_LIST_RAW="0"
    fi
  else
    GPU_LIST_RAW="0"
  fi
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST_RAW"
NUM_GPUS="${#GPU_IDS[@]}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "ERROR: No GPUs available for generation"
  exit 1
fi

MAX_PARALLEL="${MAX_PARALLEL:-$((NUM_GPUS * JOBS_PER_GPU))}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  MAX_PARALLEL=1
fi

echo "GPU list: ${GPU_IDS[*]} | NUM_GPUS=$NUM_GPUS | JOBS_PER_GPU=$JOBS_PER_GPU | MAX_PARALLEL=$MAX_PARALLEL"

echo "[3/6] Generating about ${GENERATE_N} proteins via ProteinAE autoencode..."
mapfile -t PDB_FILES < <(find "$DOWNLOAD_DIR" -maxdepth 1 -type f -name "*.pdb" | sort | head -n "$GENERATE_N")

if [[ ${#PDB_FILES[@]} -eq 0 ]]; then
  echo "ERROR: No downloaded PDB files found in $DOWNLOAD_DIR"
  exit 1
fi

COUNT=0
ACTIVE_JOBS=0
FAILED_JOBS=0
FAIL_LOG="$OUT_DIR/generation_failures.log"
: > "$FAIL_LOG"

for pdb in "${PDB_FILES[@]}"; do
  COUNT=$((COUNT + 1))
  GPU_INDEX=$(( (COUNT - 1) % NUM_GPUS ))
  GPU_ID="${GPU_IDS[$GPU_INDEX]}"
  echo "  - [$COUNT/${#PDB_FILES[@]}] Autoencoding $(basename "$pdb") on GPU $GPU_ID"
  (
    cd "$MODEL_DIR"
    PYTHONPATH="$MODEL_DIR:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="$GPU_ID" run_uv_py proteinfoundation/autoencode.py \
      --input_pdb "$pdb" \
      --output_dir "$GENERATED_RAW_DIR" \
      --mode autoencode \
      --config_path "$MODEL_DIR/configs" \
      --config_name inference_proteinae
  ) || {
    echo "$pdb" >> "$FAIL_LOG"
    exit 1
  } &

  ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
  if [[ "$ACTIVE_JOBS" -ge "$MAX_PARALLEL" ]]; then
    if wait -n; then
      ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
    else
      FAILED_JOBS=$((FAILED_JOBS + 1))
      ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
    fi
  fi
done

while [[ "$ACTIVE_JOBS" -gt 0 ]]; do
  if wait -n; then
    ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
  else
    FAILED_JOBS=$((FAILED_JOBS + 1))
    ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
  fi
done

if [[ "$FAILED_JOBS" -gt 0 ]]; then
  echo "WARN: ${FAILED_JOBS} generation jobs failed. See: $FAIL_LOG"
fi

echo "[4/6] Collecting generated sample PDBs..."
find "$GENERATED_RAW_DIR" -type f -name "sample.pdb" | while read -r sample; do
  parent="$(basename "$(dirname "$sample")")"
  cp "$sample" "$GENERATED_FLAT_DIR/${parent}.pdb"
done

GEN_COUNT=$(find "$GENERATED_FLAT_DIR" -maxdepth 1 -type f -name "*.pdb" | wc -l | tr -d ' ')
if [[ "$GEN_COUNT" -eq 0 ]]; then
  echo "ERROR: No generated sample PDB files found."
  exit 1
fi

echo "Generated PDB count: $GEN_COUNT"

echo "[5/6] Validating radius of gyration (protein-verification pipeline)..."
run_uv_py "$GPML_DIR/protein-verification/gyr_pred.py" \
  --input_dir "$DOWNLOAD_DIR" \
  --label downloaded \
  --output_csv "$VALIDATION_DIR/rg_downloaded.csv"

run_uv_py "$GPML_DIR/protein-verification/gyr_pred.py" \
  --input_dir "$GENERATED_FLAT_DIR" \
  --label generated \
  --output_csv "$VALIDATION_DIR/rg_generated.csv"

echo "[6/6] Plotting and saving Rg comparisons..."
run_uv_py - <<'PY'
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path(os.environ["OUT_DIR"])
val_dir = root / "validation"
plots_dir = root / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

df_d = pd.read_csv(val_dir / "rg_downloaded.csv")
df_g = pd.read_csv(val_dir / "rg_generated.csv")

df = pd.concat([df_d, df_g], ignore_index=True)
df.to_csv(val_dir / "rg_combined.csv", index=False)

# Plot 1: histogram overlay
plt.figure(figsize=(10, 6))
plt.hist(df_d["rg"], bins=30, alpha=0.6, label="downloaded", density=True)
plt.hist(df_g["rg"], bins=30, alpha=0.6, label="generated", density=True)
plt.xlabel("Radius of Gyration (Å)")
plt.ylabel("Density")
plt.title("Rg Distribution: Downloaded vs Generated")
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / "rg_hist_downloaded_vs_generated.png", dpi=180)
plt.close()

# Plot 2: boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([df_d["rg"], df_g["rg"]], labels=["downloaded", "generated"])
plt.ylabel("Radius of Gyration (Å)")
plt.title("Rg Summary by Method")
plt.tight_layout()
plt.savefig(plots_dir / "rg_boxplot_downloaded_vs_generated.png", dpi=180)
plt.close()

# Plot 3: sorted line comparison (first n)
n = min(len(df_d), len(df_g), 50)
d_sorted = df_d["rg"].sort_values().reset_index(drop=True).iloc[:n]
g_sorted = df_g["rg"].sort_values().reset_index(drop=True).iloc[:n]

plt.figure(figsize=(10, 6))
plt.plot(range(n), d_sorted, marker='o', linewidth=1.5, label='downloaded (sorted)')
plt.plot(range(n), g_sorted, marker='s', linewidth=1.5, label='generated (sorted)')
plt.xlabel("Sample Index (sorted)")
plt.ylabel("Radius of Gyration (Å)")
plt.title("Rg Sorted Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / "rg_sorted_comparison.png", dpi=180)
plt.close()

summary = pd.DataFrame(
    {
        "dataset": ["downloaded", "generated"],
        "count": [len(df_d), len(df_g)],
        "mean_rg": [df_d["rg"].mean(), df_g["rg"].mean()],
        "std_rg": [df_d["rg"].std(), df_g["rg"].std()],
        "min_rg": [df_d["rg"].min(), df_g["rg"].min()],
        "max_rg": [df_d["rg"].max(), df_g["rg"].max()],
    }
)
summary.to_csv(val_dir / "rg_summary_stats.csv", index=False)
print("Saved plots:")
for p in sorted(plots_dir.glob("*.png")):
    print(f" - {p}")
print("Saved summary stats:", val_dir / "rg_summary_stats.csv")
PY

echo "Done. Outputs are in: $OUT_DIR"
echo "  - Data dir: $DATA_DIR"
echo "  - Results dir: $RESULTS_DIR"
echo "  - Validation CSVs: $VALIDATION_DIR"
echo "  - Plots: $PLOTS_DIR"
