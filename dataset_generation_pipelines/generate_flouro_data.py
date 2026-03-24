import os
import random
import requests
import pandas as pd
from Bio.PDB import PDBList
import time

# 1. Setup constants
BRIGHTNESS_THRESHOLD = 0.0  # Lowered to get more data
TARGET_COUNT = 250
SAVE_DIR = "fp_pdbs"
CSV_NAME = "protein_metadata.csv"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. Query with pagination
query = """
query($cursor: String) {
  allProteins(first: 100, after: $cursor) {
    pageInfo {
      hasNextPage
      endCursor
    }
    edges {
      node {
        name
        pdb
        seq
        states {
          brightness
        }
      }
    }
  }
}
"""

url = "https://www.fpbase.org/graphql/"
raw_proteins = []
has_next = True
cursor = None

print("Fetching data from FPBase...")
while has_next:
    variables = {"cursor": cursor} if cursor else {}
    response = requests.post(url, json={"query": query, "variables": variables})
    data = response.json()
    
    if 'errors' in data or 'data' not in data:
        print("Error in API response:", data.get('errors', 'Unknown error'))
        break
        
    connection = data['data']['allProteins']
    edges = connection['edges']
    raw_proteins.extend([edge['node'] for edge in edges])
    
    page_info = connection['pageInfo']
    has_next = page_info['hasNextPage']
    cursor = page_info['endCursor']
    time.sleep(0.5)

print(f"Fetched {len(raw_proteins)} total proteins from FPbase.")

def extract_pdb_ids(pdb_value):
    if isinstance(pdb_value, str):
        return [pdb_value.strip().upper()] if pdb_value.strip() else []
    if isinstance(pdb_value, list):
        cleaned = []
        for value in pdb_value:
            if isinstance(value, str) and value.strip():
                cleaned.append(value.strip().upper())
        return cleaned
    return []

# 4. Filter for those with PDB IDs and calculate max brightness
proteins_with_pdb = []
for p in raw_proteins:
    pdb_ids = extract_pdb_ids(p.get('pdb'))
    if pdb_ids and p.get('states'):
        b_vals = [s['brightness'] for s in p['states'] if s['brightness'] is not None]
        max_b = max(b_vals) if b_vals else 0
        
        if max_b >= BRIGHTNESS_THRESHOLD:
            p['max_brightness'] = max_b
            p['pdb_ids'] = pdb_ids
            # We treat each PDB ID as a separate entry to reach higher counts
            for pid in pdb_ids:
                p_copy = p.copy()
                p_copy['single_pdb_id'] = pid
                proteins_with_pdb.append(p_copy)

print(f"Found {len(proteins_with_pdb)} PDB entries associated with these proteins.")

# 5. Random Sampling
# Make sure we don't request more than what we have
unique_pdbs = list({p['single_pdb_id']: p for p in proteins_with_pdb}.values())

if len(unique_pdbs) < TARGET_COUNT:
    print(f"Warning: Only {len(unique_pdbs)} unique PDBs available, taking all.")
    sampled = unique_pdbs
else:
    sampled = random.sample(unique_pdbs, TARGET_COUNT)

# 6. Download and Metadata Logging
pdbl = PDBList(verbose=False)
metadata = []

print(f"Downloading {len(sampled)} PDBs...")
for i, p in enumerate(sampled):
    pid = p['single_pdb_id']
    try:
        # Download
        # The return value is the path if successful
        file_path = pdbl.retrieve_pdb_file(pid, pdir=SAVE_DIR, file_format='pdb')
        
        # Log for CSV if path exists
        if os.path.exists(file_path):
            metadata.append({
                'name': p['name'],
                'pdb_id': pid,
                'brightness': p['max_brightness'],
                'sequence': p['seq']
            })
            if (i+1) % 10 == 0:
                print(f"Downloaded {i+1}/{len(sampled)} PDBs...")
    except Exception as e:
        print(f"Failed to download {pid}: {e}")

# 7. Save metadata for your regression baseline
df = pd.DataFrame(metadata)
df.to_csv(CSV_NAME, index=False)
print(f"\nDone. Metadata saved to {CSV_NAME} with {len(df)} records.")
