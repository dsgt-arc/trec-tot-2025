import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import json

# Load the classifier
MODEL_NAME = "davanstrien/ModernBERT-web-topics-1m"
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device)

def classify_queries_jsonl(jsonl_path, output_path, batch_size=1):
    """
    Classify queries from a JSONL file with fields 'query_id' and 'query'.
    """
    # Load JSONL
    queries = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            queries.append(obj)
    print(f"Loaded {len(queries)} queries from {jsonl_path}")

    input_texts = [q['query'] for q in queries]

    # Batched inference
    all_predictions = []
    for i in tqdm(range(0, len(input_texts), batch_size), desc="Running classification"):
        batch = input_texts[i:i+batch_size]
        preds = classifier(batch, truncation=True, top_k=1)
        all_predictions.extend(preds)

    # Zip results back to query_id and query
    results = []
    for idx, pred in enumerate(all_predictions):
        results.append({
            "query_id": queries[idx]["query_id"],
            "query": queries[idx]["query"],
            "predicted_topic": pred[0]["label"],
            "confidence": round(pred[0]["score"], 4)
        })

    # print a summary of how many queries were classified into each topic
    topic_counts = pd.Series([res['predicted_topic'] for res in results]).value_counts()
    print("Topic classification summary:")
    print(topic_counts)

    # Save results as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    jsonl_path = "scratch/trec-tot-2025/trec-tot-2025/user/ritesh/test-2025-queries.jsonl"  # Update with your test set path
    output_path = "scratch/trec-tot-2025/trec-tot-2025/user/ritesh/classified_test_set.csv"
    classify_queries_jsonl(jsonl_path, output_path)