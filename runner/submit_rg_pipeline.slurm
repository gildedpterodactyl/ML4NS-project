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
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
GPML_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$GPML_DIR"

mkdir -p runner/data runner/results runner/results/logs runner/results/validation runner/results/plots

export DOWNLOAD_N=300
export TARGETS="20,50,70"
export OPTIMIZE_N=100
export OPTIMIZE_STEPS=50
export JOBS_PER_GPU=1
export MAX_PARALLEL=4
export MAX_STRUCTURE_BYTES=0
export CPU_FALLBACK_ON_OOM=1
export DATA_DIR="$GPML_DIR/runner/data"
export RESULTS_DIR="$GPML_DIR/runner/results"
export UV_PYTHON_VERSION=3.11
export UV_VENV_DIR="$GPML_DIR/runner/.venv"
export TORCH_VARIANT=cu121
export INSTALL_DEPS=1

bash "$SCRIPT_DIR/run_rg_pipeline.sh"
