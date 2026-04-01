import numpy as np

def rg_from_pdb_manual(filename):
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # PDB format uses fixed columns for coordinates
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    
    coords = np.array(coords)
    
    # Center of Geometry
    center = np.mean(coords, axis=0)
    
    # Root Mean Square Distance
    rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))
    return rg

print(f"Rg (Center of Geometry): {rg_from_pdb_manual('./protein_dataset/1GAV.pdb'):.3f} Å")