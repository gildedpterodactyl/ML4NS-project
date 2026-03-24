import sys
import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path

# Add root directory to path to allow imports
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

from proteinfoundation.autoencode import load_config, ProteinAutoEncoder

def main(args):
    # Load metadata
    if not os.path.exists(args.metadata_csv):
        print(f"Metadata file not found: {args.metadata_csv}")
        return
        
    metadata_df = pd.read_csv(args.metadata_csv)

    print("Loading ProteinAE model...")
    config_dir = os.path.join(root, "configs")
    cfg = load_config(config_dir, "inference_proteinae")
    ckpt_path = Path(args.checkpoint)
    
    if not ckpt_path.exists():
        # Quick fallback if the path provided is just relatively evaluated wrong
        alt_ckpt_path = Path(root) / args.checkpoint
        if not alt_ckpt_path.exists():
            print(f"Checkpoint not found at {ckpt_path} or {alt_ckpt_path}")
            return
        ckpt_path = alt_ckpt_path
        
    try:
        model_wrapper = ProteinAutoEncoder.from_checkpoint(cfg, ckpt_path)
    except Exception as e:
        print(f"Failed to load model from checkpoint: {e}")
        return
        
    model_wrapper.model.eval()
    
    latent_vectors = []
    properties = []
    
    print(f"Processing {len(metadata_df)} proteins...")
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        pdb_id = row['pdb_id']
        property_value = row[args.property_column]
        
        pdb_filename = f"pdb{pdb_id.lower()}.ent"
        pdb_path = os.path.join(args.pdb_dir, pdb_filename)
        
        if not os.path.exists(pdb_path):
            # Fallback for different PDB naming conventions
            pdb_path_alt = os.path.join(args.pdb_dir, f"{pdb_id.lower()}.pdb")
            if not os.path.exists(pdb_path_alt):
                print(f"Warning: PDB file for {pdb_id} not found at {pdb_path} or {pdb_path_alt}. Skipping.")
                continue
            pdb_path = pdb_path_alt

        try:
            with torch.no_grad():
                # `encode` takes a Path object directly
                latent_z = model_wrapper.encode(Path(pdb_path))
            
            # Flatten latent tensor to 1D and save
            latent_z = latent_z.squeeze().mean(dim=0)
            latent_vectors.append(latent_z.cpu().numpy().flatten())
            properties.append(property_value)
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")

    if not latent_vectors:
        print("No proteins were successfully processed. Exiting.")
        return

    # Create and save dataframe
    latent_df = pd.DataFrame(latent_vectors)
    latent_df[args.property_column] = properties
    
    output_path = os.path.join(args.output_dir, f"latent_dataset_{args.property_column}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    latent_df.to_csv(output_path, index=False)
    
    print(f"Successfully created latent dataset with {len(latent_df)} entries.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract latent vectors from PDBs and pair with properties.")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to the metadata CSV (e.g., protein_metadata.csv).")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing downloaded PDB files.")
    parser.add_argument("--property_column", type=str, required=True, help="Name of the property column in the CSV (e.g., 'brightness' or 'tm_value').")
    parser.add_argument("--checkpoint", type=str, default="lightning_logs/3u249jze/checkpoints/last.ckpt", help="Path to the model checkpoint file.")
    parser.add_argument("--output_dir", type=str, default="latent_datasets", help="Directory to save the output latent dataset.")
    
    args = parser.parse_args()
    main(args)
