import os
import requests
import pandas as pd
from Bio.PDB import PDBList
import random

# 1. Setup constants
TARGET_COUNT = 250  # Number of proteins to download
SAVE_DIR = "thermo_pdbs"
CSV_NAME = "thermo_protein_metadata.csv"
API_URL = "https://fireprotdb.com/api/proteins/"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. Query FireProtDB API
print("Querying FireProtDB for proteins with Tm values...")
try:
    response = requests.get(API_URL)
    response.raise_for_status()  # Raise an exception for bad status codes
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error querying FireProtDB API: {e}")
    exit()

# 3. Filter for entries with PDB ID and Tm value
proteins_with_tm = []
for protein in data:
    if protein.get('pdb_id') and protein.get('tm_value'):
        # Some tm_values are lists, take the first one if so
        tm = protein['tm_value']
        if isinstance(tm, list):
            tm = tm[0]
        
        try:
            # Ensure Tm is a valid float
            protein['tm_value_float'] = float(tm)
            proteins_with_tm.append(protein)
        except (ValueError, TypeError):
            continue # Skip if Tm is not a valid number

print(f"Found {len(proteins_with_tm)} proteins with PDB IDs and valid Tm values.")

# 4. Random Sampling
if not proteins_with_tm:
    print("No suitable proteins found. Exiting.")
    exit()

if len(proteins_with_tm) < TARGET_COUNT:
    sampled = proteins_with_tm
else:
    sampled = random.sample(proteins_with_tm, TARGET_COUNT)

# 5. Download PDBs and Log Metadata
pdbl = PDBList()
metadata = []
downloaded_pdb_ids = set()

print(f"Attempting to download {len(sampled)} PDB files...")
for p in sampled:
    pdb_id = p['pdb_id'].strip().upper()
    if not pdb_id or pdb_id in downloaded_pdb_ids:
        continue

    try:
        # Download the PDB file
        pdbl.retrieve_pdb_file(pdb_id, pdir=SAVE_DIR, file_format='pdb')
        
        # Log metadata for CSV
        metadata.append({
            'name': p.get('name', 'N/A'),
            'pdb_id': pdb_id,
            'tm_value': p['tm_value_float'],
            'uniprot_id': p.get('uniprot_id', 'N/A')
        })
        downloaded_pdb_ids.add(pdb_id)
        print(f"Successfully downloaded PDB for {pdb_id}")
    except Exception as e:
        print(f"Failed to download PDB for {pdb_id}: {e}")

# 6. Save metadata
if metadata:
    df = pd.DataFrame(metadata)
    df.to_csv(CSV_NAME, index=False)
    print(f"\nDone. Metadata for {len(df)} proteins saved to {CSV_NAME}")
else:
    print("\nNo PDB files were successfully downloaded. No metadata saved.")