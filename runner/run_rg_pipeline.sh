#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPML_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$GPML_DIR/.." && pwd)"
cd "$GPML_DIR"

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.11}"
UV_VENV_DIR="${UV_VENV_DIR:-$SCRIPT_DIR/.venv}"
TORCH_VARIANT="${TORCH_VARIANT:-cu121}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
AUTOENCODE_ACCELERATOR="${AUTOENCODE_ACCELERATOR:-gpu}"

DOWNLOAD_N="${DOWNLOAD_N:-300}"
TARGETS="${TARGETS:-20,50,70}"
OPTIMIZE_N="${OPTIMIZE_N:-100}"
OPTIMIZE_STEPS="${OPTIMIZE_STEPS:-50}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
CPU_FALLBACK_ON_OOM="${CPU_FALLBACK_ON_OOM:-1}"
MAX_STRUCTURE_BYTES="${MAX_STRUCTURE_BYTES:-0}"

DOWNLOAD_DIR="$DATA_DIR/downloaded_300_pdbs"
DOWNLOAD_META="$DATA_DIR/downloaded_300_properties.json"
OPT_ROOT="$RESULTS_DIR/optimization"
TARGETS_ROOT="$RESULTS_DIR/targets"
PLOTS_DIR="$RESULTS_DIR/plots"
MODEL_DIR="$GPML_DIR/model"

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$OPT_ROOT" "$TARGETS_ROOT" "$PLOTS_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed."
  exit 1
fi

echo "[0/8] Setting up uv environment and dependencies..."
if [[ ! -d "$UV_VENV_DIR" ]]; then
  uv venv --python "$UV_PYTHON_VERSION" "$UV_VENV_DIR"
fi
UV_PYTHON_BIN="$UV_VENV_DIR/bin/python"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  if [[ "$TORCH_VARIANT" == "cu121" ]]; then
    uv pip install --python "$UV_PYTHON_BIN" --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
    PYG_WHL_URL="${PYG_WHL_URL:-https://data.pyg.org/whl/torch-2.4.0+cu121.html}"
  else
    uv pip install --python "$UV_PYTHON_BIN" --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu
    PYG_WHL_URL="${PYG_WHL_URL:-https://data.pyg.org/whl/torch-2.4.0+cpu.html}"
  fi

  uv pip install --python "$UV_PYTHON_BIN" requests numpy pandas matplotlib biopython scipy lightning torch-geometric einops omegaconf hydra-core loguru python-dotenv jaxtyping dm-tree cpdb-protein wget biopandas transformers timm
  uv pip install --python "$UV_PYTHON_BIN" --find-links "$PYG_WHL_URL" torch_scatter torch_sparse torch_cluster
fi

run_uv_py() {
  uv run --python "$UV_PYTHON_BIN" "$@"
}

target_tag() {
  local v="$1"
  if [[ "$v" == *.* ]]; then
    echo "${v//./p}"
  else
    echo "$v"
  fi
}

run_autoencode() {
  local input_path="$1"
  local output_dir="$2"
  local mode="$3"
  local gpu_id="$4"
  shift 4

  (
    cd "$MODEL_DIR"
    if [[ "$gpu_id" == "cpu" ]]; then
      PYTHONPATH="$MODEL_DIR:${PYTHONPATH:-}" AUTOENCODE_ACCELERATOR="cpu" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" run_uv_py proteinfoundation/autoencode.py \
        --input_pdb "$input_path" \
        --output_dir "$output_dir" \
        --mode "$mode" \
        --config_path "$MODEL_DIR/configs" \
        --config_name inference_proteinae "$@"
    else
      CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH="$MODEL_DIR:${PYTHONPATH:-}" AUTOENCODE_ACCELERATOR="$AUTOENCODE_ACCELERATOR" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" run_uv_py proteinfoundation/autoencode.py \
        --input_pdb "$input_path" \
        --output_dir "$output_dir" \
        --mode "$mode" \
        --config_path "$MODEL_DIR/configs" \
        --config_name inference_proteinae "$@"
    fi
  )
}

split_latent_tensor() {
  local latent_pt="$1"
  local output_root="$2"

  run_uv_py - "$latent_pt" "$output_root" <<'PY'
import sys
from pathlib import Path
import torch

latent_pt = Path(sys.argv[1])
output_root = Path(sys.argv[2])
output_root.mkdir(parents=True, exist_ok=True)

z = torch.load(latent_pt, map_location="cpu")
if z.ndim == 1:
    z = z.unsqueeze(0)

for i, row in enumerate(z, start=1):
    d = output_root / f"{i:04d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(row.detach().cpu(), d / "latent_repr.pt")
PY
}

decode_latent_tensor() {
  local latent_pt="$1"
  local input_root="$2"
  local output_root="$3"
  local gpu_id="$4"

  rm -rf "$input_root" "$output_root"
  mkdir -p "$input_root" "$output_root"
  split_latent_tensor "$latent_pt" "$input_root"

  for latent_dir in "$input_root"/*; do
    [[ -d "$latent_dir" ]] || continue
    run_autoencode "$latent_dir" "$output_root" decode "$gpu_id"
  done
}

flatten_generated_pdbs() {
  local source_root="$1"
  local flat_root="$2"
  rm -rf "$flat_root"
  mkdir -p "$flat_root"
  find "$source_root" -type f -name "sample.pdb" | while read -r sample; do
    parent="$(basename "$(dirname "$sample")")"
    cp "$sample" "$flat_root/${parent}.pdb"
  done
}

echo "[1/8] Resolving model checkpoint..."
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

echo "[2/8] Resolving GPU resources..."
GPU_LIST_RAW="${GPU_LIST:-}"
if [[ "$AUTOENCODE_ACCELERATOR" == "cpu" ]]; then
  GPU_IDS=("cpu")
  NUM_GPUS=1
else
  if [[ -z "$GPU_LIST_RAW" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      GPU_LIST_RAW="$CUDA_VISIBLE_DEVICES"
    elif command -v nvidia-smi >/dev/null 2>&1; then
      GPU_COUNT_DETECTED="$(nvidia-smi -L | wc -l | tr -d ' ')"
      if [[ "$GPU_COUNT_DETECTED" -gt 0 ]]; then
        GPU_LIST_RAW="$(seq -s, 0 $((GPU_COUNT_DETECTED - 1)))"
      else
        echo "ERROR: No GPU detected"
        exit 1
      fi
    else
      echo "ERROR: No GPU detected"
      exit 1
    fi
  fi
  IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST_RAW"
  NUM_GPUS="${#GPU_IDS[@]}"
fi
MAX_PARALLEL="${MAX_PARALLEL:-$((NUM_GPUS * JOBS_PER_GPU))}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  MAX_PARALLEL=1
fi
echo "Device list: ${GPU_IDS[*]} | NUM_GPUS=$NUM_GPUS | MAX_PARALLEL=$MAX_PARALLEL"

echo "[3/8] Downloading and VRAM-filtering proteins to exactly ${DOWNLOAD_N}..."
run_uv_py "$SCRIPT_DIR/download_vram_filtered.py" \
  --target-count "$DOWNLOAD_N" \
  --save-dir "$DOWNLOAD_DIR" \
  --metadata-file "$DOWNLOAD_META" \
  --checkpoint "$FOUND_CKPT" \
  --project-root "$MODEL_DIR" \
  --device cuda

echo "[4/8] Training Rg oracle from filtered proteins..."
TRAINER_SCRIPT="$GPML_DIR/regression/train_tm_from_latent.py"
if [[ ! -f "$TRAINER_SCRIPT" ]]; then
  TRAINER_SCRIPT="$ROOT_DIR/regression/train_tm_from_latent.py"
fi
if [[ ! -f "$TRAINER_SCRIPT" ]]; then
  echo "ERROR: Missing train_tm_from_latent.py"
  exit 1
fi

ORACLE_MODEL="$OPT_ROOT/ridge_latent_radius_of_gyration_model.npz"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" run_uv_py "$TRAINER_SCRIPT" \
  --checkpoint "$FOUND_CKPT" \
  --metadata "$DOWNLOAD_META" \
  --metadata-format json \
  --target-column radius_of_gyration \
  --max-structure-bytes "$MAX_STRUCTURE_BYTES" \
  --cpu-fallback-on-oom "$CPU_FALLBACK_ON_OOM" \
  --project-root "$MODEL_DIR" \
  --output-dir "$OPT_ROOT" \
  --output-name "$(basename "$ORACLE_MODEL")" \
  --device auto

echo "[5/8] Running multi-target optimization (${TARGETS})..."
IFS=',' read -r -a TARGET_LIST <<< "$TARGETS"
opt_pids=()
for idx in "${!TARGET_LIST[@]}"; do
  t="${TARGET_LIST[$idx]}"
  t="${t// /}"
  tag="$(target_tag "$t")"
  t_dir="$TARGETS_ROOT/target_${tag}"
  opt_dir="$t_dir/optimization"
  mkdir -p "$opt_dir"

  gpu="${GPU_IDS[$((idx % NUM_GPUS))]}"
  (
    if [[ "$gpu" == "cpu" ]]; then
      PYTHONPATH="$GPML_DIR:${PYTHONPATH:-}" run_uv_py "$GPML_DIR/optimization/experiment_runner.py" \
        --checkpoint "$ORACLE_MODEL" \
        --target-tm "$t" \
        --n-vectors "$OPTIMIZE_N" \
        --n-steps "$OPTIMIZE_STEPS" \
        --output-dir "$opt_dir"
    else
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$GPML_DIR:${PYTHONPATH:-}" run_uv_py "$GPML_DIR/optimization/experiment_runner.py" \
        --checkpoint "$ORACLE_MODEL" \
        --target-tm "$t" \
        --n-vectors "$OPTIMIZE_N" \
        --n-steps "$OPTIMIZE_STEPS" \
        --output-dir "$opt_dir"
    fi
  ) &
  opt_pids+=("$!")
done

for pid in "${opt_pids[@]}"; do
  wait "$pid"
done

echo "[6/8] Decoding generated latents for all targets..."
decode_pids=()
for idx in "${!TARGET_LIST[@]}"; do
  t="${TARGET_LIST[$idx]}"
  t="${t// /}"
  tag="$(target_tag "$t")"
  t_dir="$TARGETS_ROOT/target_${tag}"
  opt_dir="$t_dir/optimization"

  grad_input="$DATA_DIR/latent_inputs/target_${tag}/gradient"
  ess_input="$DATA_DIR/latent_inputs/target_${tag}/slice"
  grad_raw="$t_dir/generated_gradient_raw"
  ess_raw="$t_dir/generated_slice_raw"
  grad_flat="$t_dir/generated_gradient_pdbs"
  ess_flat="$t_dir/generated_slice_pdbs"

  gpu="${GPU_IDS[$((idx % NUM_GPUS))]}"
  (
    decode_latent_tensor "$opt_dir/z_final_gradient_ascent.pt" "$grad_input" "$grad_raw" "$gpu"
    decode_latent_tensor "$opt_dir/z_final_ess.pt" "$ess_input" "$ess_raw" "$gpu"
    flatten_generated_pdbs "$grad_raw" "$grad_flat"
    flatten_generated_pdbs "$ess_raw" "$ess_flat"
  ) &
  decode_pids+=("$!")
done

for pid in "${decode_pids[@]}"; do
  wait "$pid"
done

echo "[7/8] Validating Rg for downloaded and generated sets..."
run_uv_py "$GPML_DIR/protein-verification/gyr_pred.py" \
  --input_dir "$DOWNLOAD_DIR" \
  --label downloaded \
  --output_csv "$RESULTS_DIR/validation/rg_downloaded.csv"

for t in "${TARGET_LIST[@]}"; do
  t="${t// /}"
  tag="$(target_tag "$t")"
  t_dir="$TARGETS_ROOT/target_${tag}"
  val_dir="$t_dir/validation"
  mkdir -p "$val_dir"

  run_uv_py "$GPML_DIR/protein-verification/gyr_pred.py" \
    --input_dir "$t_dir/generated_gradient_pdbs" \
    --label "gradient_ascent_t${tag}" \
    --output_csv "$val_dir/rg_gradient_ascent.csv"

  run_uv_py "$GPML_DIR/protein-verification/gyr_pred.py" \
    --input_dir "$t_dir/generated_slice_pdbs" \
    --label "slice_sampling_t${tag}" \
    --output_csv "$val_dir/rg_slice_sampling.csv"
done

echo "[8/8] Plotting per-target histograms and violins..."
run_uv_py "$SCRIPT_DIR/plot_multi_target_rg.py" \
  --targets-root "$TARGETS_ROOT" \
  --downloaded-csv "$RESULTS_DIR/validation/rg_downloaded.csv" \
  --output-dir "$PLOTS_DIR" \
  --targets "$TARGETS"

echo "Done."
echo "  Downloaded proteins: $DOWNLOAD_DIR"
echo "  Target outputs: $TARGETS_ROOT"
echo "  Plots: $PLOTS_DIR"
