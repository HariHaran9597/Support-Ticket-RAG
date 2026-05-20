import pandas as pd
import random
import os

def generate_eval_set(input_file="data/processed/tickets.csv", output_file="eval/eval_questions.csv"):
    df = pd.read_csv(input_file)
    sample_df = df.sample(25, random_state=42)
    
    eval_data = []
    for _, row in sample_df.iterrows():
        eval_data.append({
            "question": row['customer_issue'],
            "expected_ticket_id": row['ticket_id'],
            "expected_category": row['category']
        })
        
    # Add unanswerable questions
    eval_data.append({
        "question": "What is the recipe for chocolate cake?",
        "expected_ticket_id": None,
        "expected_category": None
    })
    eval_data.append({
        "question": "Who won the World Cup in 2022?",
        "expected_ticket_id": None,
        "expected_category": None
    })
    eval_data.append({
        "question": "How do I bake a chocolate chip cookie?",
        "expected_ticket_id": None,
        "expected_category": None
    })
    
    df_eval = pd.DataFrame(eval_data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_eval.to_csv(output_file, index=False)
    print(f"Generated eval set with {len(df_eval)} questions.")

if __name__ == "__main__":
    generate_eval_set()
