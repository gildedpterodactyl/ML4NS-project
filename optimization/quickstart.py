#!/usr/bin/env python3
"""
Quick Start Script for the Oracle Hack Challenge

This script runs a complete end-to-end example with small parameters to validate
that everything is working correctly.

Usage:
    python optimization/quickstart.py

Expected runtime: ~2 minutes
Output: optimization/outputs_quickstart/
"""

from pathlib import Path
import subprocess
import sys


def main():
    workspace_root = Path("/home/rohitj/Documents/courses/MLNS")
    
    print("=" * 70)
    print("OPTIMIZATION PIPELINE: QUICKSTART")
    print("=" * 70)
    
    checkpoint = workspace_root / "regression/outputs/ridge_latent_tm_model.npz"
    if not checkpoint.exists():
        print(f"\n❌ ERROR: Oracle checkpoint not found at {checkpoint}")
        print("\nFirst, train the oracle:")
        print(f"  cd {workspace_root}")
        print("  python regression/train_tm_from_latent.py \\")
        print("      --checkpoint ae_r1_d8_v1.ckpt \\")
        print("      --metadata fireprot_metadata.csv \\")
        print("      --structures-dir fireprot_structures")
        sys.exit(1)
    
    print(f"\n✓ Found Oracle: {checkpoint}")
    
    output_dir = workspace_root / "optimization/outputs_quickstart"
    
    print(f"\nRunning optimization on 5 vectors (20 steps each)...")
    print("This should take ~2 minutes...\n")
    
    cmd = [
        "python",
        str(workspace_root / "optimization/experiment_runner.py"),
        "--checkpoint", str(checkpoint),
        "--target-tm", "65.0",
        "--n-vectors", "5",
        "--n-steps", "20",
        "--gradient-lr", "0.1",
        "--gradient-penalty", "0.05",
        "--output-dir", str(output_dir),
        "--seed", "42",
    ]
    
    result = subprocess.run(cmd, cwd=workspace_root)
    
    if result.returncode != 0:
        print(f"\n❌ Error running experiment")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ SUCCESS!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nKey outputs:")
    print(f"  - trajectories.csv (convergence curves)")
    print(f"  - z_final_gradient_ascent.pt (optimized vectors)")
    print(f"  - z_final_ess.pt (sampled vectors)")
    print(f"  - optimization_stats.json (summary)")
    
    print("\nNext steps:")
    print("  1. Plot trajectories: python optimization/plot_trajectories.py")
    print("  2. Full run: python optimization/experiment_runner.py --n-vectors 100")
    print("  3. Validate: python validation/decode_and_validate.py \\")
    print("               --z-vectors optimization/outputs_quickstart/z_final_ess.pt")


if __name__ == "__main__":
    main()
