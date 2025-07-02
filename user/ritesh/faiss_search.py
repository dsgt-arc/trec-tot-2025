import os
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import json
from collections import defaultdict
import gc

def search_shards_efficient(output_dir, queries_jsonl_path, model_name, top_k=5, start_shard=0, tmp_results_dir=None):
    """
    Memory-efficient search across all shards using queries from JSONL
    """
    # 1. Load queries from JSONL
    queries = []
    query_ids = []
    with open(queries_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            queries.append(data['query'])  # Assuming 'text' is the query field
            query_ids.append(data['query_id'])  # Store the query ID
    
    # 2. Load model and encode queries (keep in memory)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    query_embeddings = model.encode(queries)

    # normalize the query embeddings
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    
    # 3. Get all shard files
    shard_files = list_shards(output_dir)
    
    # 4. Initialize results storage using query IDs
    all_results = {query_id: [] for query_id in query_ids}
    
    # 5. Process each shard
    for i, shard in enumerate(shard_files):
        if i < start_shard:
            continue  # Skip until start_shard
        print(f"Processing shard {i+1}/{len(shard_files)}: {shard['filename']}")
        
        # Load shard
        df = pd.read_parquet(shard['filepath'])
        embeddings = df[[col for col in df.columns if col.startswith('emb_')]].values
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build FAISS index for this shard
        dimension = embeddings.shape[1]
        if faiss.get_num_gpus() > 0 and False:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(dimension))
        else:
            index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32').copy())
        
        # clear the embeddings from memory
        del embeddings
        
        # Search ALL queries at once
        scores, indices = index.search(query_embeddings.astype('float32').copy(), top_k)
        
        # clear the index from memory
        del index

        # Store results for all queries using query IDs
        for query_idx, query_id in enumerate(query_ids):
            for score, idx in zip(scores[query_idx], indices[query_idx]):
                all_results[query_id].append({
                    'metadata': df.iloc[idx][['id', 'title']].to_dict(),
                    'score': float(score),
                    'shard': shard['filename']
                })
        
        # Clear shard from memory
        del scores, indices, df
        gc.collect()
        
        shard_result_path = os.path.join(tmp_results_dir, f"results_{shard['filename']}.jsonl")
        with open(shard_result_path, "w", encoding="utf-8") as f:
            for query_id, results in all_results.items():
                for result in results:
                    out = {"query_id": query_id, **result}
                    f.write(json.dumps(out) + "\n")
        # Clear in-memory results
        all_results = {query_id: [] for query_id in query_ids}
    

def list_shards(output_dir):
    """
    List all created shards
    """
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist")
        return []
    
    shard_files = []
    for file in os.listdir(output_dir):
        if file.endswith('.parquet') and file.startswith('shard_'):
            filepath = os.path.join(output_dir, file)
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            shard_files.append({
                'filename': file,
                'filepath': filepath,
                'size_mb': size_mb
            })
    
    # Sort by shard number
    shard_files.sort(key=lambda x: int(x['filename'].split('_')[1]))
    
    print(f"\nFound {len(shard_files)} shards:")
    total_size = 0
    for shard in shard_files:
        print(f"  {shard['filename']} ({shard['size_mb']:.1f} MB)")
        total_size += shard['size_mb']
    
    print(f"Total size: {total_size:.1f} MB")
    return shard_files


def aggregate_shard_results(tmp_results_dir, top_k=5, output_file=None):
    """
    Aggregate per-shard results from temp files and return top-k per query.
    """
    final_results = defaultdict(list)

    # Read all result files
    for fname in os.listdir(tmp_results_dir):
        if fname.endswith(".jsonl"):
            with open(os.path.join(tmp_results_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    query_id = obj["query_id"]
                    final_results[query_id].append(obj)

    # For each query, sort and keep top-k
    top_k_results = {}
    for query_id, results in final_results.items():
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        top_k_results[query_id] = sorted_results

    # write the results to a jsonl file
    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            for query_id, results in top_k_results.items():
                for result in results:
                    f.write(json.dumps(result) + "\n")

    return top_k_results, output_file


if __name__ == "__main__":
    output_dir = "/workspace/embeddings_shards_from_parquet"
    queries_jsonl_path = "/workspace/dev1-2025-queries.jsonl"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 1000
    start_shard = 0
    tmp_results_dir = "/workspace/tmp_search_results"
    os.makedirs(tmp_results_dir, exist_ok=True)
    search_shards_efficient(output_dir, queries_jsonl_path, model_name, top_k, start_shard, tmp_results_dir)
    
    output_file = "/workspace/top_k_results.jsonl"  # or .json if you prefer
    results, output_file = aggregate_shard_results(tmp_results_dir, top_k, output_file)
    # print the results
    for query_id, results in results.items():
        print(f"Query ID: {query_id}")
        for result in results:
            print(f"  Result: {result['metadata']}")
            print(f"    Score: {result['score']}")
            print(f"    Shard: {result['shard']}")
    print(f"Results written to {output_file}")

