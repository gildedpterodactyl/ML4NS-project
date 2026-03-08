import argparse
import torch
from graphein.protein.tensor.io import protein_to_pyg
from proteinfoundation.guidance.oracles import RadiusOfGyration, ContactDensity, HBondScore, ClashScore

def main(args):
    # Dictionary mapping oracle names to classes
    oracles = {
        "rg": RadiusOfGyration(),
        "contact_density": ContactDensity(),
        "hbond": HBondScore(),
        "clash": ClashScore(),
    }

    for pdb_path in args.pdb_files:
        print(f"Analyzing {pdb_path}...")
        try:
            # Convert PDB to PyG graph representation
            graph = protein_to_pyg(
                pdb_path=pdb_path,
                chain_selection="A", # Assuming a single chain for simplicity
                device=torch.device("cpu")
            )

            # Extract coordinates and mask
            coords = graph.coords.unsqueeze(0)
            mask = graph.residue_type.bool().unsqueeze(0)

            # Calculate and print properties from all oracles
            for name, oracle in oracles.items():
                value = oracle(coords, mask).item()
                print(f"  - {name}: {value:.4f}")

        except Exception as e:
            print(f"  Could not process {pdb_path}. Error: {e}")
        print("-" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify geometric properties of generated protein PDB files.")
    parser.add_argument(
        "pdb_files",
        type=str,
        nargs='+',
        help="Path(s) to the PDB file(s) to verify.",
    )
    args = parser.parse_args()
    main(args)
