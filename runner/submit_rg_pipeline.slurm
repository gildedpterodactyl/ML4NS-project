#!/usr/bin/env bash
#SBATCH --job-name=rg_pipeline
#SBATCH --output=runner/results/logs/rg_pipeline_%j.out
#SBATCH --error=runner/results/logs/rg_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
GPML_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$GPML_DIR/.." && pwd)"

cd "$GPML_DIR"
mkdir -p runner/data runner/results runner/results/logs runner/results/validation runner/results/plots

# Optional: activate your env here
# source "$ROOT_DIR/datasets/.venv/bin/activate"

# Tuning knobs for the runner
export DOWNLOAD_N=200
export GENERATE_N=50
export JOBS_PER_GPU=1
export MAX_PARALLEL=4
export DATA_DIR="$GPML_DIR/runner/data"
export RESULTS_DIR="$GPML_DIR/runner/results"
export UV_PYTHON_VERSION=3.11
export UV_VENV_DIR="$GPML_DIR/runner/.venv"
export TORCH_VARIANT=cu121

# If your cluster masks GPU IDs, CUDA_VISIBLE_DEVICES is already set by Slurm.
# You can override manually if needed:
# export GPU_LIST="0,1,2,3"

bash "$SCRIPT_DIR/run_rg_pipeline.sh"
