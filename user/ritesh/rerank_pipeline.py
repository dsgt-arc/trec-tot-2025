import pyterrier as pt
import pandas as pd
import json
from typing import List, Dict
import numpy as np
import os
import torch
from sentence_transformers import CrossEncoder
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize PyTerrier
if not pt.started():
    pt.init()

def load_queries_from_jsonl(file_path: str) -> pd.DataFrame:
    """
    Load queries from JSONL file
    """
    queries = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            queries.append({
                'qid': str(data['query_id']),
                'query': data['query']
            })
    
    return pd.DataFrame(queries)

def load_results_from_jsonl(file_path: str) -> pd.DataFrame:
    """
    Load results from JSONL file and convert to PyTerrier format
    """
    results = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            results.append({
                'qid': data['query_id'],
                'docno': data['metadata']['id'],
                'title': data['metadata']['title'],
                'score': data['score'],
                'shard': data['shard']
            })
    
    return pd.DataFrame(results)

def load_required_texts_from_shards(results_df, shard_dir):
    final_records = []

    for shard_name, shard_df in results_df.groupby("shard"):
        shard_name = shard_name.split("_emb")[0]
        shard_name = shard_name + ".parquet"
        shard_path = os.path.join(shard_dir, shard_name)
        doc_ids_needed = set(shard_df["docno"])

        # Load parquet shard (batch processable)
        shard_parquet = pd.read_parquet(shard_path)

        # Filter only required doc_ids
        filtered = shard_parquet[shard_parquet["id"].astype(str).isin(doc_ids_needed)]

        # Add (shard + doc_id) → text mapping
        doc_text_map = dict(zip(filtered["id"].astype(str), filtered["text"]))

        # Assign back text
        for i, row in shard_df.iterrows():
            doc_text = doc_text_map.get(row["docno"], "")
            final_records.append({**row, "text": doc_text})

    return pd.DataFrame(final_records)


def save_results_to_jsonl(df, score_column, output_file):
    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            result = {
                "query_id": str(row["qid"]),
                "score": float(row[score_column]),
                "shard": row["shard"],
                "metadata": {
                    "id": row["docno"],
                    "title": row["title"]
                }
            }
            f.write(json.dumps(result) + "\n")

def rerank_with_model(final_df, model_name, score_column, batch_size=8, device="cuda"):
    reranker = CrossEncoder(model_name, device=device)
    reranked = reranker.predict(list(zip(final_df["query"], final_df["text"])), batch_size=batch_size)

    # save the score back to the final_df
    final_df[score_column] = reranked

    return final_df

if __name__ == "__main__":
    # File paths
    queries_file = "/workspace/dev1-2025-queries.jsonl"
    results_file = "miniLM_dev1_top_k_results.jsonl"
    shard_dir = "/workspace/split_parquet_shards"
    output_dir = "/root/trec-tot-2025/user/ritesh/results"
    models = [
        ("cross-encoder/ms-marco-MiniLM-L6-v2", "miniLM_dev1_top_k_results_reranked_minilm_l6.jsonl"),
        ("BAAI/bge-reranker-base", "miniLM_dev1_top_k_results_reranked_bge_base.jsonl")
    ]

    # queries_df = load_queries_from_jsonl(queries_file)
    # results_df = load_results_from_jsonl(results_file)
    # results_df = load_required_texts_from_shards(results_df, shard_dir)
    # final_df = results_df.merge(queries_df, on="qid")
    # # save final_records to parquet
    # final_df.to_parquet("/root/trec-tot-2025/user/ritesh/results/miniLM_dev1_top_k_results_with_corpus_text.parquet")

    final_df = pd.read_parquet("/root/trec-tot-2025/user/ritesh/results/miniLM_dev1_top_k_results_with_corpus_text.parquet")

    batch_size = 512
    for model_name, output_file in models:
        print(f"Reranking using: {model_name}")
        start_time = time.time()
        reranked_df = rerank_with_model(final_df, model_name, score_column=f"{model_name}_reranked_score", batch_size=batch_size)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        save_results_to_jsonl(reranked_df, score_column=f"{model_name}_reranked_score", output_file=os.path.join(output_dir, output_file))
        print(f"Saved reranked results to {output_file}\n")