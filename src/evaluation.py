import pandas as pd
import time
import os
import sys

# Add parent directory to path to allow importing src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag import RAGPipeline

def evaluate(eval_file="eval/eval_questions.csv", results_file="eval/results.csv"):
    if not os.path.exists(eval_file):
        print(f"Eval file {eval_file} not found.")
        return
        
    df_eval = pd.read_csv(eval_file)
    rag = RAGPipeline()
    
    results = []
    
    for _, row in df_eval.iterrows():
        question = row['question']
        expected_ticket = str(row['expected_ticket_id'])
        is_unanswerable = pd.isna(row['expected_ticket_id']) or row['expected_ticket_id'] == 'None'
        
        # Retrieve top 5 for hit@5
        retrieved = rag.retrieve(question, top_k=5)
        retrieved_ids = [str(r['ticket_id']) for r in retrieved]
        top_similarity = retrieved[0]['distance'] if retrieved else None
        
        hit_at_3 = 1 if expected_ticket in retrieved_ids[:3] else 0
        hit_at_5 = 1 if expected_ticket in retrieved_ids[:5] else 0
        
        # Generate answer
        answer_res = rag.generate_answer(question, top_k=3)
        status = answer_res['status']
        
        if is_unanswerable:
            refusal_correct = 1 if status == 'refused' else 0
        else:
            refusal_correct = 1 if status == 'answered' else 0
            
        results.append({
            "question": question,
            "expected_ticket": expected_ticket,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "top_similarity": top_similarity,
            "status": status,
            "refusal_correct": refusal_correct,
            "latency": answer_res['latency'],
            "answer": answer_res['answer']
        })
        
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    df_results.to_csv(results_file, index=False)
    
    print("--- Evaluation Summary ---")
    print(f"Hit@3: {df_results['hit_at_3'].mean():.2f}")
    print(f"Hit@5: {df_results['hit_at_5'].mean():.2f}")
    print(f"Avg Top Similarity: {df_results['top_similarity'].mean():.2f}")
    print(f"Refusal Accuracy: {df_results['refusal_correct'].mean():.2f}")
    print(f"Avg Latency: {df_results['latency'].mean():.2f}s")
    
if __name__ == "__main__":
    evaluate()
