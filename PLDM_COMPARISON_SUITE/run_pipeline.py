#!/usr/bin/env python3
"""
Run the complete PLDM comparison suite pipeline.
This script replaces the bash orchestration and works with or without uv.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None, description=""):
    """Run a shell command and report output."""
    if description:
        print(f"\n{'='*80}")
        print(f"[PIPELINE] {description}")
        print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}: {cmd}")
        return False

def main():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    print(f"[PIPELINE] Starting PLDM Comparison Suite")
    print(f"[PIPELINE] Working directory: {script_dir}")
    
    # Set environment variables
    n = os.environ.get("N", "50")
    device = os.environ.get("DEVICE", "cpu")
    with_structure = os.environ.get("WITH_STRUCTURE", "false")
    
    print(f"[PIPELINE] Configuration: N={n}, DEVICE={device}, WITH_STRUCTURE={with_structure}")
    
    commands = [
        ("python generate_sequences.py --mode baseline --proldm-root ../PROLDM_OUTLIER --train-csv data/mut_data/GFP-train.csv --baseline-ckpt train_logs/GFP/dropout_tiny_epoch_1000.pt --dataset GFP --n 50 --omega 20.0 --device cpu --out-dir baseline/outputs", "baseline", "BASELINE: Generating sequences"),
        ("python generate_sequences.py --mode ess --proldm-root ../PROLDM_OUTLIER --train-csv data/mut_data/GFP-train.csv --ae-ckpt train_logs/GFP/epoch_1000.pt --dataset GFP --n 50 --num-chains 8 --burnin 20 --max-steps 5000 --alpha 0.5 --latent-temperature 0.75 --esm-weight 0.5 --reg-weight 0.5 --tess-delta 0.05 --device cpu --out-dir ess/outputs", "ess", "ESS: Generating sequences"),
        ("python generate_sequences.py --mode tess --proldm-root ../PROLDM_OUTLIER --train-csv data/mut_data/GFP-train.csv --ae-ckpt train_logs/GFP/epoch_1000.pt --dataset GFP --n 50 --num-chains 8 --burnin 20 --max-steps 5000 --tess-delta 0.14 --alpha 0.5 --latent-temperature 0.75 --esm-weight 0.5 --reg-weight 0.5 --device cpu --out-dir tess/outputs", "tess", "TESS: Generating sequences"),
        ("python generate_sequences.py --mode transport_ess --proldm-root ../PROLDM_OUTLIER --train-csv data/mut_data/GFP-train.csv --ae-ckpt train_logs/GFP/epoch_1000.pt --dataset GFP --n 50 --num-chains 8 --burnin 20 --max-steps 5000 --tess-delta 0.14 --alpha 0.5 --latent-temperature 0.75 --esm-weight 0.5 --reg-weight 0.5 --transport-strength 0.5 --device cpu --out-dir transport/outputs", "transport_ess", "TRANSPORT ESS: Generating sequences"),
    ]
    
    # Create output directories
    for method in ["baseline", "ess", "tess", "transport"]:
        Path(f"{method}/outputs").mkdir(parents=True, exist_ok=True)
        Path(f"{method}/results").mkdir(parents=True, exist_ok=True)
    Path("common/results").mkdir(parents=True, exist_ok=True)
    
    # Run all generation commands
    failed = False
    for cmd, method, desc in commands:
        if not run_command(cmd, description=desc):
            print(f"[ERROR] Failed at {method}")
            failed = True
            # Don't stop, try to continue
    
    if not failed:
        print("\n[PIPELINE] All generations completed successfully")
    else:
        print("\n[WARNING] Some generations failed but continuing...")
    
    # Run evaluations
    print("\n[PIPELINE] Running evaluations...")
    
    eval_commands = [
        ("python evaluate_method.py --method-name baseline_pldm --input-csv baseline/outputs/baseline_pldm.csv --results-csv baseline/results/results.csv --proldm-root ../PROLDM_OUTLIER --train-csv data/mut_data/GFP-train.csv --ae-ckpt train_logs/GFP/epoch_1000.pt --dataset GFP --device cpu", "BASELINE: Evaluating"),
        ("python evaluate_method.py --method-name ess --input-csv ess/outputs/results_ess.csv --results-csv ess/results/results.csv --proldm-root ../PROLDM_OUTLIER --train-csv data/mut_data/GFP-train.csv --ae-ckpt train_logs/GFP/epoch_1000.pt --dataset GFP --device cpu", "ESS: Evaluating"),
        ("python evaluate_method.py --method-name tess --input-csv tess/outputs/results_tess.csv --results-csv tess/results/results.csv --proldm-root ../PROLDM_OUTLIER --train-csv data/mut_data/GFP-train.csv --ae-ckpt train_logs/GFP/epoch_1000.pt --dataset GFP --device cpu", "TESS: Evaluating"),
        ("python evaluate_method.py --method-name transport_ess --input-csv transport/outputs/results_transport_ess.csv --results-csv transport/results/results.csv --proldm-root ../PROLDM_OUTLIER --train-csv data/mut_data/GFP-train.csv --ae-ckpt train_logs/GFP/epoch_1000.pt --dataset GFP --device cpu", "TRANSPORT ESS: Evaluating"),
    ]
    
    for cmd, desc in eval_commands:
        run_command(cmd, description=desc)
    
    # Run plotting
    print("\n[PIPELINE] Generating plots...")
    run_command("python merge_and_plot.py", description="PLOTTING: Generating comparison plots")
    
    print("\n[PIPELINE] Pipeline completed!")
    print("[PIPELINE] Results available in:")
    for method in ["baseline", "ess", "tess", "transport"]:
        print(f"  - {method}/results/results.csv")

if __name__ == "__main__":
    main()
