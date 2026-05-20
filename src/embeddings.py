import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def create_embeddings(input_file="data/processed/chunks.json", output_file="data/processed/embeddings.npy"):
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [chunk['text'] for chunk in chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, embeddings)
    print(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    create_embeddings()
