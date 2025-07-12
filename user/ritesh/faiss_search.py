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
import glob

def search_shards_efficient(output_dir, queries_jsonl_path, model_name, top_k=5, start_shard=0, tmp_results_dir=None, batch_size=512):
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
    
    # 2. Load model and detect max token length
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    max_tokens = model.tokenizer.model_max_length

    # For each query, chunk and mean-pool to handle long queries
    query_embeddings = []
    for query in queries:
        emb = get_full_text_embedding(query, model, max_tokens=max_tokens, stride=max_tokens//2, batch_size=batch_size)
        query_embeddings.append(emb)
    query_embeddings = np.stack(query_embeddings)

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


def chunk_text(text, tokenizer, max_tokens=512, stride=512):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        if len(chunk_tokens) < max_tokens:
            break
    return chunks

def get_full_text_embedding(text, model, max_tokens=512, stride=512, batch_size=512):
    tokenizer = model.tokenizer
    chunks = chunk_text(text, tokenizer, max_tokens=max_tokens, stride=stride)
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=batch_size)
    return np.mean(embeddings, axis=0)


def aggregate_shard_results_batched(tmp_results_dir, top_k=5, batch_size=10, output_file=None):
    """
    Aggregate results in batches to manage memory usage.
    Process batch_size files at a time, get top-k, then move to next batch.
    """
    # Get all result files
    result_files = glob.glob(os.path.join(tmp_results_dir, "*.jsonl"))
    result_files.sort()  # Process in order
    
    # Initialize final results storage
    final_results = defaultdict(list)
    
    # Process files in batches
    for i in range(0, len(result_files), batch_size):
        batch_files = result_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: files {i+1}-{min(i+batch_size, len(result_files))}")
        
        # Collect results from this batch
        batch_results = defaultdict(list)
        
        for fname in batch_files:
            print(f"  Reading {os.path.basename(fname)}...")
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    query_id = obj["query_id"]
                    batch_results[query_id].append(obj)
        
        # Merge batch results with final results
        for query_id, results in batch_results.items():
            final_results[query_id].extend(results)
        
        # Get top-k for each query after this batch
        for query_id in final_results:
            sorted_results = sorted(final_results[query_id], key=lambda x: x["score"], reverse=True)
            final_results[query_id] = sorted_results[:top_k]
        
        # Clear batch results to free memory
        del batch_results
        gc.collect()
        
    # write the results to a jsonl file
    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            for query_id, results in final_results.items():
                for result in results:
                    f.write(json.dumps(result) + "\n")
    
    return final_results


if __name__ == "__main__":
    output_dir = "/workspace/embeddings_shards_from_parquet"
    queries_jsonl_path = "/workspace/dev3-2025-queries.jsonl"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size = 512
    top_k = 1000
    start_shard = 0
    tmp_results_dir = "/workspace/tmp_search_results"
    os.makedirs(tmp_results_dir, exist_ok=True)
    # search_shards_efficient(output_dir, queries_jsonl_path, model_name, top_k, start_shard, tmp_results_dir, batch_size)
    
    output_file = "/workspace/miniLM_dev3_top_k_results.jsonl"  # or .json if you prefer
    # results, output_file = aggregate_shard_results(tmp_results_dir, top_k, output_file)
    results = aggregate_shard_results_batched(tmp_results_dir, top_k, 10, output_file)
    # print the results
    for query_id, results in results.items():
        print(f"Query ID: {query_id}")
        for result in results:
            print(f"  Result: {result['metadata']}")
            print(f"    Score: {result['score']}")
            print(f"    Shard: {result['shard']}")
    print(f"Results written to {output_file}")

