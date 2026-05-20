import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import time

load_dotenv()

class RAGPipeline:
    def __init__(self, index_file="artifacts/faiss.index", metadata_file="artifacts/metadata.parquet"):
        self.index = faiss.read_index(index_file)
        self.metadata = pd.read_parquet(metadata_file)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it in your .env file or Streamlit secrets.")
        self.client = Groq(api_key=api_key)
        self.model = "qwen/qwen3-32b"
        
    def retrieve(self, query, top_k=3):
        query_emb = self.encoder.encode([query])
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            meta = self.metadata.iloc[idx].to_dict()
            results.append({
                "ticket_id": meta["ticket_id"],
                "category": meta["category"],
                "text": meta["text"],
                "distance": float(dist)
            })
        return results

    def generate_answer(self, query, top_k=3, distance_threshold=1.5):
        start_time = time.time()
        retrieved = self.retrieve(query, top_k)
        
        if not retrieved or retrieved[0]['distance'] > distance_threshold:
            return {
                "answer": "I don't have enough evidence in the ticket database to answer this question.",
                "citations": [],
                "retrieved": retrieved,
                "latency": time.time() - start_time,
                "status": "refused"
            }
            
        context = ""
        for doc in retrieved:
            context += f"\n--- Ticket {doc['ticket_id']} (Category: {doc['category']}) ---\n{doc['text']}\n"
            
        system_prompt = (
            "You are a helpful customer support assistant. You MUST answer the user's question "
            "ONLY using the provided ticket context. If the context does not contain the answer, "
            "say 'I don't have enough evidence in the ticket database.' "
            "When you answer, you MUST cite the ticket IDs you used. "
            "For example: 'To reset your password, click the link (Citation: TICKET_1234)'"
        )
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                model=self.model,
                temperature=0.0
            )
            answer = chat_completion.choices[0].message.content
            status = "answered"
            citations = [doc['ticket_id'] for doc in retrieved]
        except Exception as e:
            answer = f"Error calling LLM: {str(e)}"
            status = "error"
            citations = []
            
        return {
            "answer": answer,
            "citations": citations,
            "retrieved": retrieved,
            "latency": time.time() - start_time,
            "status": status
        }

if __name__ == "__main__":
    rag = RAGPipeline()
    res = rag.generate_answer("How do I cancel my order?")
    print(res['answer'])
