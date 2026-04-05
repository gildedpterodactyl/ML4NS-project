import os
import requests
import json
import time
import numpy as np
from Bio.PDB import PDBParser

# --- Configuration ---
UNIQUE_TARGET = 50
SAVE_DIR = "protein_dataset"
METADATA_FILE = "protein_properties.json"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_BASE_URL = "https://files.rcsb.org/download/"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- Step 1: Search (Your Working Code) ---
query = {
    "query": {
        "type": "terminal",
        "service": "full_text",
        "parameters": {"value": "protein"}
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {"start": 0, "rows": UNIQUE_TARGET},
        "sort": [{"sort_by": "score", "direction": "desc"}]
    }
}

print("Querying RCSB...")
response = requests.post(SEARCH_URL, json=query, timeout=30)
pdb_ids = [item['identifier'] for item in response.json().get('result_set', [])] if response.status_code == 200 else []

# --- Step 2: Property Calculation Function ---
def calculate_rg(pdb_path):
    """Calculates Radius of Gyration for CA atoms."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        # Extract only CA coordinates
        ca_coords = [res['CA'].get_coord() for model in structure for chain in model for res in chain if 'CA' in res]
        
        if not ca_coords: return None
        
        coords = np.array(ca_coords)
        center_of_mass = np.mean(coords, axis=0)
        # Rg formula: sqrt( mean( squared distances from center ) )
        rg = np.sqrt(np.mean(np.sum((coords - center_of_mass)**2, axis=1)))
        return float(rg)
    except Exception:
        return None

# --- Step 3: Download & Mapping Loop ---
property_map = {}

for i, pdb_id in enumerate(pdb_ids):
    pdb_id = pdb_id.upper()
    file_path = os.path.join(SAVE_DIR, f"{pdb_id}.pdb")
    
    print(f"[{i+1}/{len(pdb_ids)}] Processing {pdb_id}...", end="\r")
    
    # Download the PDB file
    r = requests.get(f"{DOWNLOAD_BASE_URL}{pdb_id}.pdb")
    if r.status_code == 200:
        with open(file_path, 'w') as f:
            f.write(r.text)
        
        # Calculate the property (Compactness / Rg)
        rg_value = calculate_rg(file_path)
        
        if rg_value:
            property_map[pdb_id] = {
                "radius_of_gyration": round(rg_value, 3),
                "local_path": file_path
            }
        else:
            # If we can't calculate Rg (e.g. no CA atoms), remove the file
            os.remove(file_path)
    
    time.sleep(0.1) # Be polite to RCSB servers

# --- Step 4: Save the JSON Map ---
with open(METADATA_FILE, 'w') as f:
    json.dump(property_map, f, indent=4)

print(f"\n\nDone! {len(property_map)} proteins downloaded and mapped to {METADATA_FILE}")