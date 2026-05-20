import json
import numpy as np
import faiss
import os
import pandas as pd

def build_vectorstore(
    embeddings_file="data/processed/embeddings.npy", 
    chunks_file="data/processed/chunks.json",
    index_file="artifacts/faiss.index",
    metadata_file="artifacts/metadata.parquet"
):
    embeddings = np.load(embeddings_file)
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    faiss.write_index(index, index_file)
    
    df_meta = pd.DataFrame([c['metadata'] for c in chunks])
    df_meta['text'] = [c['text'] for c in chunks]
    df_meta.to_parquet(metadata_file, index=False)
    
    print(f"Built FAISS index with {index.ntotal} vectors and saved to {index_file}")

if __name__ == "__main__":
    build_vectorstore()
