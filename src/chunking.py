import pandas as pd
import json
import os

def chunk_tickets(input_file="data/processed/tickets.csv", output_file="data/processed/chunks.json"):
    df = pd.read_csv(input_file)
    chunks = []
    for _, row in df.iterrows():
        # Tickets are small enough to be single chunks
        chunk = {
            "id": row['ticket_id'],
            "text": row['source_text'],
            "metadata": {
                "ticket_id": row['ticket_id'],
                "category": row['category'],
                "customer_issue": row['customer_issue']
            }
        }
        chunks.append(chunk)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
    print(f"Created {len(chunks)} chunks and saved to {output_file}")

if __name__ == "__main__":
    chunk_tickets()
