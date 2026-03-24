#!/bin/bash

# ProSteer Full Pipeline Execution
# This script runs the entire workflow for data generation, latent space extraction,
# and property regression model training for both fluorescence and thermostability.

# Stop on first error
# set -e

# Define the correct python executable
PYTHON_EXEC="/home/vishak/miniforge3/envs/proteinae/bin/python"

echo "======================================================"
echo "STEP 1: DATA GENERATION"
echo "======================================================"

# Run fluorescence data generation
echo "--- Generating fluorescence dataset (from FPbase) ---"
$PYTHON_EXEC dataset_generation_pipelines/generate_flouro_data.py
echo "--- Fluorescence dataset generation complete. ---"

# Run thermostability data generation
echo "--- Generating thermostability dataset (from FireProtDB) ---"
$PYTHON_EXEC dataset_generation_pipelines/generate_thermo_data.py
echo "--- Thermostability dataset generation complete. ---"


echo "======================================================"
echo "STEP 2: LATENT VECTOR EXTRACTION"
echo "======================================================"

# Extract latents for fluorescence data
echo "--- Extracting latent vectors for fluorescence ---"
$PYTHON_EXEC scripts/extract_latents.py --checkpoint checkpoints/ae_r1_d8_v1.ckpt \
    --metadata_csv protein_metadata.csv \
    --pdb_dir fp_pdbs/ \
    --property_column brightness \
    --output_dir latent_datasets

echo "--- Fluorescence latent extraction complete. ---"

# Extract latents for thermostability data
echo "--- Extracting latent vectors for thermostability ---"
$PYTHON_EXEC scripts/extract_latents.py --checkpoint checkpoints/ae_r1_d8_v1.ckpt \
    --metadata_csv thermo_protein_metadata.csv \
    --pdb_dir thermo_pdbs/ \
    --property_column tm_value \
    --output_dir latent_datasets

echo "--- Thermostability latent extraction complete. ---"


echo "======================================================"
echo "STEP 3: REGRESSION MODEL TRAINING"
echo "======================================================"

# Train regressor for brightness
echo "--- Training regression model for brightness ---"
$PYTHON_EXEC scripts/train_regressor.py \
    --latent_csv latent_datasets/latent_dataset_brightness.csv \
    --property_column brightness \
    --output_dir models

echo "--- Brightness regressor training complete. ---"

# Train regressor for thermostability
echo "--- Training regression model for tm_value (thermostability) ---"
$PYTHON_EXEC scripts/train_regressor.py \
    --latent_csv latent_datasets/latent_dataset_tm_value.csv \
    --property_column tm_value \
    --output_dir models

echo "--- Thermostability regressor training complete. ---"

echo "======================================================"
echo "PIPELINE FINISHED SUCCESSFULLY!"
echo "======================================================"
echo "Next steps:"
echo "1. Check the 'models' directory for the trained regressors ('regressor_brightness.pt', 'regressor_tm_value.pt')."
echo "2. Check the 'latent_datasets' directory for the generated datasets."
echo "3. Check 'fp_pdbs/' and 'thermo_pdbs/' for downloaded PDB files."
echo "4. Check '*.csv' files for the metadata."

