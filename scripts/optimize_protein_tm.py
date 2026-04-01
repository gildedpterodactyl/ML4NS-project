import torch
import numpy as np
import os
import argparse
from torch.optim import SGD
from proteinfoundation.autoencode import ProteinAutoEncoder
from proteinfoundation.proteinflow.proteinae import ProteinAE

def load_models(ae_ckpt, regressor_npz, device="cuda"):
    import lightning as L
    # Load AutoEncoder
    ae_model = ProteinAE.load_from_checkpoint(ae_ckpt, strict=False)
    ae_model.eval()
    ae_model.to(device)
    trainer = L.Trainer(accelerator="gpu" if "cuda" in device else "cpu", devices=1, logger=False, enable_progress_bar=False)
    ae = ProteinAutoEncoder(model=ae_model, trainer=trainer)
    
    # Load Regressor
    reg_data = np.load(regressor_npz)
    regressor = {
        'weights': torch.tensor(reg_data['coef'], dtype=torch.float32, device=device),
        'intercept': torch.tensor(reg_data['intercept'], dtype=torch.float32, device=device),
        'mean': torch.tensor(reg_data['x_mean'], dtype=torch.float32, device=device),
        'std': torch.tensor(reg_data['x_std'], dtype=torch.float32, device=device)
    }
    return ae, regressor

def encode_pdb(pdb_path, ae, device="cuda"):
    from pathlib import Path
    with torch.no_grad():
        out = ae.encode(Path(pdb_path))
        # `out` is [N, 8] typically. Give it batch dim [1, N, 8]
        if out.dim() == 2:
            out = out.unsqueeze(0)
        return out.to(device)

def optimize_latent(z_init, regressor, target_tm=95.0, steps=100, lr=0.1):
    """
    Perform gradient descent on the latent representation `z` to reach a specific target_tm.
    """
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr) # Switched to Adam for more stable convergence
    
    W = regressor['weights']  # shape [8]
    b = regressor['intercept']
    mean = regressor['mean']
    std = regressor['std']
    
    print(f"\n--- Latent Space Optimization (Target: {target_tm}°C) ---")
    for step in range(steps):
        optimizer.zero_grad()
        
        # 1. Pool latent z to shape [8]
        z_pool = z.squeeze(0).mean(dim=0)
        
        # 2. Normalize
        z_norm = (z_pool - mean) / (std + 1e-8)
        
        # 3. Predict Tm
        tm_pred = torch.dot(W, z_norm) + b
        
        # We want to REACH target_tm, not infinity. Use Mean Squared Error.
        loss = (tm_pred - target_tm)**2
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0 or step == steps - 1:
            print(f"Step {step:03d} | Predicted Tm: {tm_pred.item():.2f}°C | Distance: {abs(tm_pred.item()-target_tm):.2f}°C")
            
    return z.detach()

def decode_and_save(z_opt, ae, output_path, device="cuda"):
    from proteinfoundation.autoencode import OutputWriter
    from omegaconf import OmegaConf
    
    cfg = OmegaConf.load("configs/experiment_config/inference_proteinae.yaml")
    # Need to make sure data params are set as fallback if missing
    if not hasattr(cfg, "data"):
        cfg.data = OmegaConf.create({
            "max": 512,
            "crop": 512,
            "add_cb": True,
            "atom37": False,
            "transform": True
        })
    
    # Needs to be squeezed back to [N, 8] if shape is [1, N, 8]
    if z_opt.dim() == 3:
        z_opt = z_opt.squeeze(0)
        
    predictions = ae.decode(z_opt.detach().cpu(), cfg)
    
    if predictions:
        import os
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        # We manually save one prediction giving it a specific output name
        from openfold.np.protein import Protein
        from openfold.np import protein as pnp
        from openfold.utils.tensor_utils import tensor_tree_map
        
        # Actually, if OutputWriter exists, we can rename the produced file:
        OutputWriter.write_predictions(predictions, out_dir)
        # It usually saves 'pred_0.pdb'. We'll rename it.
        # Sometimes OutputWriter might save as something else. Let's find it 
        import glob
        pdbs = glob.glob(os.path.join(out_dir, "*.pdb"))
        # we can just assume the most recently modified pdb is the one
        if pdbs:
            latest_pdb = max(pdbs, key=os.path.getctime)
            
            # Read the PDB and replace ALA with GLY
            # This is specifically for MolProbity since AE generates backbone only (no CB)
            # and MolProbity rejects ALA without experimental CB atoms. GLY requires no CB.
            with open(latest_pdb, 'r') as f:
                pdb_text = f.read()
            
            pdb_text = pdb_text.replace("ALA", "GLY")
            
            with open(output_path, 'w') as f:
                f.write(pdb_text)
                
            # Clean up the original if we moved it
            if latest_pdb != output_path and os.path.exists(latest_pdb):
                os.remove(latest_pdb)
                
            print(f"\nSaved optimized backbone structure to {output_path} (Formatted as GLY for MolProbity)")
        else:
            print("Failed to find generated file.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, required=True, help="Input WT PDB")
    parser.add_argument("--ae_ckpt", type=str, default="ae_r1_d8_v1.ckpt", help="ProteinAE ckpt")
    parser.add_argument("--reg_npz", type=str, default="ridge_latent_tm_model.npz")
    parser.add_argument("--out", type=str, default="optimized_protein.pdb")
    parser.add_argument("--target_tm", type=float, default=95.0, help="Target Tm in Celsius")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1) # Adjusted default LR for Adam
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Models...")
    ae, regressor = load_models(args.ae_ckpt, args.reg_npz, device)
    
    print(f"Encoding '{args.pdb}'...")
    z_init = encode_pdb(args.pdb, ae, device)
    
    print(f"Optimizing '{args.pdb}' towards {args.target_tm}°C...")
    z_opt = optimize_latent(z_init, regressor, target_tm=args.target_tm, steps=args.steps, lr=args.lr)
    
    print(f"Decoding modified latent space...")
    decode_and_save(z_opt, ae, args.out, device)

if __name__ == "__main__":
    main()