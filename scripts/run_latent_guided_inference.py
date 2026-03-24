#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Vishak
# SPDX-License-Identifier: MIT

import os
import sys

# Assume script run from ProteinAE_v1 root
sys.path.append(os.getcwd())

import torch
import numpy as np
from torch.utils.data import DataLoader
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from proteinfoundation.inference_ldm import GenDataset
from proteinfoundation.proteinflow.proteinae_ldm import ProteinLDM
from proteinfoundation.guidance.oracles import OracleRegistry

def run_guided_inference():
    print("Initializing environment...")
    torch.set_grad_enabled(False)
    
    # We load standard PLDM config
    GlobalHydra.instance().clear()
    initialize(config_path="../configs/experiment_config", version_base="1.3")
    cfg = compose(config_name="inference_ucond_pldm_200M_512")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load PLDM checkpoint
    # For now, let's just make sure the config is loaded.
    ckpt_path = "checkpoints/pldm_200M.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Please provide a mock or real checkpoint.")
        return

    print("Loading PLDM model...")
    try:
        if os.path.getsize(ckpt_path) > 0:
            model = ProteinLDM.load_from_checkpoint(ckpt_path).eval().to(device)
        else:
            raise ValueError("Checkpoint file is 0 bytes")
    except Exception as e:
        print(f"Warning: Could not load real checkpoint ({e}). Initializing model from config randomly for pipeline test...")
        model = ProteinLDM(cfg).eval().to(device)
        
    model.configure_inference(cfg, nn_ag=None)
    
    print("Setting up Latent Regressor Oracle...")
    # Load Brightness Regressor
    model_path = "models/regressor_brightness.pt"
    if not os.path.exists(model_path):
         print(f"Error: Regressor model not found at {model_path}")
         return
         
    # Our target brightness could be a normalized value or high value
    oracle = OracleRegistry.get(
        "latent_brightness", 
        target=1.0, 
        direction="minimize", # Usually maximize for brightness, or minimize MSE to target
        model_path=model_path
    )
    model.guidance_oracle = oracle
    model.guidance_scale = 10.0  # Guidance scale eta
    print("Oracle ready.")

    # Simple mock dataset for 1 protein of length 100
    dataset = GenDataset(
        nres=[100], 
        nsamples=[1], 
        dt=cfg.dt, 
        dt_latent=cfg.dt_latent, 
        len_cath_codes=None
    )
    dataloader = DataLoader(dataset, batch_size=1)
    
    print("Generating Latent Guided Structure...")
    # Inference mode must evaluate with enable_grad to allow autograd for oracle!
    # Lightning's predict_step naturally wraps with no_grad, but we bypass inside our oracle via torch.enable_grad
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            output = model.predict_step(batch, batch_idx=0)
            
        coords = output["pred_coords"]
        print(f"Success! Generated shape: {coords.shape}")
        break
        
if __name__ == "__main__":
    run_guided_inference()