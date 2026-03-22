import os
import random
import requests
import pandas as pd
from Bio.PDB import PDBList

# 1. Setup constants
BRIGHTNESS_THRESHOLD = 1.5  
TARGET_COUNT = 10
SAVE_DIR = "fp_pdbs"
CSV_NAME = "protein_metadata.csv"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. Query with variables (Decimal must be passed as a variable string)
query = """
query($minBrightness: Decimal!) {
  allProteins(spectralBrightness_Gt: $minBrightness) {
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

variables = {
    "minBrightness": str(BRIGHTNESS_THRESHOLD)
}

url = "https://www.fpbase.org/graphql/"
response = requests.post(url, json={"query": query, "variables": variables})
data = response.json()

# 3. Correct Data Extraction
if 'data' not in data or not data['data']['allProteins']:
    print("Error in API response:", data.get('errors', 'Unknown error'))
    exit()


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

# Extract the list of nodes from the edges
edges = data['data']['allProteins']['edges']
raw_proteins = [edge['node'] for edge in edges]

# 4. Filter for those with PDB IDs and calculate max brightness
proteins_with_pdb = []
for p in raw_proteins:
    pdb_ids = extract_pdb_ids(p.get('pdb'))
    if pdb_ids and p.get('states'):
        # Get highest brightness from available states
        b_vals = [s['brightness'] for s in p['states'] if s['brightness'] is not None]
        max_b = max(b_vals) if b_vals else 0
        
        if max_b >= BRIGHTNESS_THRESHOLD:
            p['max_brightness'] = max_b
            p['pdb_ids'] = pdb_ids
            proteins_with_pdb.append(p)

print(f"Found {len(proteins_with_pdb)} proteins with PDB IDs.")

# 5. Random Sampling
if len(proteins_with_pdb) < TARGET_COUNT:
    sampled = proteins_with_pdb
else:
    sampled = random.sample(proteins_with_pdb, TARGET_COUNT)

# 6. Download and Metadata Logging
pdbl = PDBList()
metadata = []

for p in sampled:
    for pid in p['pdb_ids']:
        try:
            # Download
            pdbl.retrieve_pdb_file(pid, pdir=SAVE_DIR, file_format='pdb')

            # Log for CSV
            metadata.append({
                'name': p['name'],
                'pdb_id': pid,
                'brightness': p['max_brightness'],
                'sequence': p['seq']
            })
            print(f"Successfully downloaded {pid}")
        except Exception as e:
            print(f"Failed to download {pid}: {e}")

# 7. Save metadata for your regression baseline
df = pd.DataFrame(metadata)
df.to_csv(CSV_NAME, index=False)
print(f"\nDone. Metadata saved to {CSV_NAME}")