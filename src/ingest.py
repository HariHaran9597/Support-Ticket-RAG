import os
import pandas as pd
from datasets import load_dataset

def ingest_data():
    print("Loading dataset from Hugging Face...")
    # Load a public customer support dataset
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    
    # Convert to pandas dataframe
    df = dataset.to_pandas()
    
    # Take a sample for faster local processing
    df = df.sample(1000, random_state=42).reset_index(drop=True)
    
    # Create normalized columns
    df['ticket_id'] = [f"TICKET_{i:04d}" for i in range(len(df))]
    df['customer_issue'] = df['instruction']
    df['resolution'] = df['response']
    df['source_text'] = "Category: " + df['category'] + "\nIssue: " + df['customer_issue'] + "\nResolution: " + df['resolution']
    
    # Select only required columns
    df_clean = df[['ticket_id', 'customer_issue', 'category', 'resolution', 'source_text']]
    
    os.makedirs("data/processed", exist_ok=True)
    df_clean.to_csv("data/processed/tickets.csv", index=False)
    print(f"Saved {len(df_clean)} cleaned tickets to data/processed/tickets.csv")

if __name__ == "__main__":
    ingest_data()
