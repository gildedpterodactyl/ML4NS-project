import requests
import json
import os
import time
import pandas as pd

# --- Configuration ---
TM_THRESHOLD = 40
UNIQUE_TARGET = 300  # Try to realistically get all unique from 500 pool
SEARCH_LIMIT = 500  # API max is 500 per request
SAVE_DIR = "fireprot_structures"
METADATA_FILE = "fireprot_metadata.csv"
FIREPROT_API_URL = "https://loschmidt.chemi.muni.cz/fireprotdb/api/search"

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# --- 1. Query FireProtDB ---
metadata_list = []
downloaded_files = set()
offset = 0

print(f"Connecting to FireProtDB to find {UNIQUE_TARGET} unique proteins...")

while len(downloaded_files) < UNIQUE_TARGET:
    query_obj = {"tree": {"variable": "TM", "operator": "GREATER_THAN", "value": TM_THRESHOLD}}
    params = {"query": json.dumps(query_obj), "limit": SEARCH_LIMIT, "offset": offset, "format": "json"}
    
    response = requests.get(FIREPROT_API_URL, params=params)
    if response.status_code != 200:
        print(f"API Error at offset {offset}")
        break
    data = response.json()
    if not data:
        print("No more data from API.")
        break

    for entry in data:
        # Break early if we hit our target number of unique structures
        if len(downloaded_files) >= UNIQUE_TARGET:
            break

    seq_data = entry.get('sequence') or {}
    mutant_data = entry.get('mutant') or {}
    container = seq_data if seq_data.get('experiment') else mutant_data
    experiment = container.get('experiment', {})
    
    # Extract Tm
    tm_value = None
    measurements = experiment.get('measurements', [])
    for m in measurements:
        if m.get('type') == 'TM':
            tm_value = m.get('numValue')
            break
    
    if tm_value is None: continue

    # Extract ID
    pdb_id = None
    uniprot_id = None
    structures = container.get('structures', [])
    if structures:
        pdb_id = structures[0].get('wwpdb') or structures[0].get('pdbId')

    source_seq = container.get('sourceSequence') or seq_data
    links = source_seq.get('proteinLinks', [])
    for link in links:
        refs = link.get('protein', {}).get('references', [])
        for ref in refs:
            if ref.get('type') == 'UNIPROTKB':
                uniprot_id = ref.get('accession')
                break
    
    # Determine Filename and URL
    filename = None
    url = None
    if pdb_id:
        pdb_id = str(pdb_id).strip().upper()
        filename = f"{pdb_id}.pdb"
        url = f"https://files.rcsb.org/download/{filename}"
        source_type = "PDB"
    elif uniprot_id:
        uniprot_id = str(uniprot_id).strip().upper()
        filename = f"{uniprot_id}.pdb"
        source_type = "AlphaFold"
        try:
            af_api = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
            af_res = requests.get(af_api).json()
            if af_res: url = af_res[0].get('pdbUrl')
        except: url = None

    # ONLY download if we haven't seen this PDB/UniProt ID before
    if filename and url and filename not in downloaded_files:
        print(f"Progress: {len(downloaded_files)+1}/{UNIQUE_TARGET} | Fetching {filename}...", end="\r")
        try:
            r = requests.get(url)
            if r.status_code == 200:
                with open(os.path.join(SAVE_DIR, filename), 'w') as f:
                    f.write(r.text)
                metadata_list.append({
                    "id": filename.split('.')[0], 
                    "type": source_type, 
                    "tm": tm_value, 
                    "filename": filename
                })
                downloaded_files.add(filename)
                time.sleep(0.15) 
        except:
            continue
    
    # Increment offset after processing all entries in this batch
    offset += SEARCH_LIMIT

# --- 3. Save Metadata ---
if metadata_list:
    pd.DataFrame(metadata_list).to_csv(METADATA_FILE, index=False)
    print(f"\n\nSuccess! Downloaded {len(downloaded_files)} unique protein structures.")
else:
    print("\nNo unique metadata extracted.")