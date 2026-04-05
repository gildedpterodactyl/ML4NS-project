from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("BIO_LM_API_KEY")

import requests
import time

def predict_melting_temperatures(sequence_list, api_key):
    """
    Batches sequences and fetches Tm predictions from ESM2StabP.
    """
    url = "https://biolm.ai/api/v3/esm2stabp/predict/"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }
    
    tm_results = []
    
    # Process the 1,000 sequences in chunks of 8
    for i in range(0, len(sequence_list), 8):
        batch = sequence_list[i:i+8]
        
        # Format the batch according to the BioLM schema
        # You can also optionally add "growth_temp" or "experimental_condition" to these dicts
        items = [{"sequence": seq} for seq in batch]
        payload = {"items": items}
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                for res in data.get('results', []):
                    # The API returns both 'melting_temperature' and a boolean 'is_thermophilic' (Tm > 60C)
                    tm_results.append(res['melting_temperature'])
            else:
                print(f"API Error on batch {i}: {response.text}")
                # Pad with None so your results array aligns with your input array
                tm_results.extend([None] * len(batch))
                
            # Be polite to the API rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Request failed: {e}")
            tm_results.extend([None] * len(batch))
            
    return tm_results

# --- How to use it in your pipeline ---

# 1. These are the sequences you generated from ProteinMPNN
my_1000_sequences = [
    "EKAWWYNLFMNMWVFKGQINVCPQAISSYGQMCDIFTRVEEYGKMKILSICRILESYTWTGGNMDIQLKCHTWNSFFHGIYAATFEHMALMFIKDDMSKLQTFMAGKERHDTKFKVLTGEYCCIQMGRNIYVENTAEICEKLERVAQSVILWPDAWVRAQCMSWRFKLMWNQSYVLDGGYFWSLQMMAEQYAPMLAMDGF", 
    "GPLGIWMPTTDITQAVTDPWHMTRSWQYWADKWLWSIGYKLAQLGCTMSEGHQYYMMLIIVEGYGPIGPPQNHQHRSHMVCVCYDLYDRNIKEACHFEPGWHYKNFWEAFHDILHNLKLYASPTIFVCSKLYCVVGRPAKVALVFQLGENRYEYTGRVMDNEESHHAWAFFQAECHPADRRRPIYPWWQTDTESWCVQKY"
]

# 2. Run the batch predictor
temperatures = predict_melting_temperatures(my_1000_sequences, api_key)

# 3. Print or save your results
for seq, tm in zip(my_1000_sequences[:5], temperatures[:5]):
    print(f"Sequence: {seq[:10]}... | Predicted Tm: {tm:.2f} °C")