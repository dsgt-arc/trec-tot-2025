import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import pandas as pd
import time
import os
from datetime import datetime
import torch
import zstandard as zstd
from collections import defaultdict
import psutil
import itertools


def create_sharded_embeddings(jsonl_path,
                              model_name,
                              batch_size=512,
                              max_entries=None,
                              output_prefix='wikipedia',
                              shard_size=1000000,
                              output_dir='embeddings_shards',
                              start_entry=0):
    """
    Create embeddings and save as multiple smaller Parquet files
    """

    print(f"Loading model: {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    dimension = model.get_sentence_embedding_dimension()
    print(f"Model dimension: {dimension}")
    # Automatically detect max token length
    max_tokens = model.tokenizer.model_max_length
    print(f"Model max token length: {max_tokens}")

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    processed_count = 0
    shard_count = 0
    start_time = time.time()

    # Handle zstd-compressed files
    if jsonl_path.endswith('.zst'):
        file_handle = zstd.open(jsonl_path, 'rt', encoding='utf-8')
    else:
        file_handle = open(jsonl_path, 'r', encoding='utf-8')

    with file_handle as f:
        batch_texts = []
        batch_metadata = []
        current_entry = 0
        for line in tqdm(itertools.islice(f, start_entry, None), desc="Processing documents"):
            if max_entries is not None and processed_count >= max_entries:
                break
            data = json.loads(line.strip())
            if 'e5' in model_name.lower():
                text = f"passage: {data['title']} {data['text']}"
            else:
                text = f"{data['title']} {data['text']}"
            batch_texts.append(text)
            batch_metadata.append({
                'id': data['id'],
                'url': data['url'],
                'title': data['title']
            })
            if len(batch_texts) >= batch_size:
                # Batch all chunks from these docs
                pooled_embeddings = get_full_text_embeddings_batched(
                    batch_texts, model, max_tokens=max_tokens, batch_size=batch_size)
                for emb, meta in zip(pooled_embeddings, batch_metadata):
                    row = meta.copy()
                    row.update({f'emb_{k}': float(v)
                               for k, v in enumerate(emb)})
                    all_data.append(row)
                processed_count += len(batch_texts)
                if len(all_data) >= shard_size:
                    shard_count += 1
                    save_shard(all_data, shard_count, model_name, output_dir)
                    kill_other_instances()  # Kill other instances after saving a shard
                    all_data = []
                if processed_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    docs_per_second = processed_count / elapsed_time
                    print(f"Processed {processed_count:,} documents | "
                          f"Rate: {docs_per_second:.1f} docs/s | "
                          f"Elapsed: {elapsed_time/60:.1f} minutes")
                batch_texts = []
                batch_metadata = []
            current_entry += 1
        # Process remaining items
        if batch_texts:
            pooled_embeddings = get_full_text_embeddings_batched(
                batch_texts, model, max_tokens=max_tokens, batch_size=batch_size)
            for emb, meta in zip(pooled_embeddings, batch_metadata):
                row = meta.copy()
                row.update({f'emb_{k}': float(v) for k, v in enumerate(emb)})
                all_data.append(row)
            processed_count += len(batch_texts)
        if all_data:
            shard_count += 1
            save_shard(all_data, shard_count, model_name, output_dir)


def save_shard(data, shard_number, model_name, output_dir):
    """
    Save a shard of data to Parquet file
    """
    df = pd.DataFrame(data)
    model_short = model_name.split('/')[-1]

    # Create descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"shard_{shard_number:04d}_{model_short}_{len(data)}docs_{timestamp}.parquet"
    filepath = os.path.join(output_dir, filename)

    # Save to Parquet
    df.to_parquet(filepath, index=False)

    # Print shard info
    print(
        f"Saved shard {shard_number}: {len(data):,} docs | Size: {os.path.getsize(filepath)/1024/1024:.1f} MB | File: {filename}")

    return filepath


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


def search_shards(output_dir, query, model_name, top_k=5):
    """
    Search across all shards
    """
    shard_files = list_shards(output_dir)

    if not shard_files:
        print("No shards found")
        return []

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)

    # Format query
    if 'e5' in model_name.lower():
        formatted_query = f"query: {query}"
    else:
        formatted_query = query

    # Encode query
    query_embedding = model.encode([formatted_query])

    all_results = []

    # Search each shard
    for shard in shard_files:
        print(f"Searching {shard['filename']}...")

        # Load shard
        df = pd.read_parquet(shard['filepath'])

        # Get embedding columns
        emb_cols = [col for col in df.columns if col.startswith('emb_')]
        embeddings = df[emb_cols].values

        # Get metadata
        metadata_cols = ['id', 'url', 'title']
        metadata = df[metadata_cols].to_dict('records')

        # Create FAISS index for this shard
        dimension = len(emb_cols)
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(
                res, 0, faiss.IndexFlatIP(dimension))
        else:
            index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))

        # Search
        scores, indices = index.search(
            query_embedding.astype('float32'), top_k)

        # Add results
        for score, idx in zip(scores[0], indices[0]):
            all_results.append({
                'metadata': metadata[idx],
                'score': float(score),
                'shard': shard['filename']
            })

    # Sort by score and return top_k
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:top_k]


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


def get_full_text_embeddings_batched(docs, model, max_tokens=256, batch_size=512):
    tokenizer = model.tokenizer
    chunked_texts = []
    doc_indices = []  # Track which doc each chunk belongs to
    for idx, text in enumerate(docs):
        chunks = chunk_text(text, tokenizer, max_tokens=max_tokens)
        chunked_texts.extend(chunks)
        doc_indices.extend([idx] * len(chunks))
    # Batch encode all chunks
    embeddings = model.encode(
        chunked_texts, show_progress_bar=False, batch_size=batch_size)
    # Group by doc and mean-pool
    doc_to_embs = defaultdict(list)
    for emb, doc_idx in zip(embeddings, doc_indices):
        doc_to_embs[doc_idx].append(emb)
    pooled = [np.mean(doc_to_embs[i], axis=0) for i in range(len(docs))]
    return pooled


def kill_other_instances():
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] != current_pid and 'embeddings.py' in ' '.join(proc.info['cmdline']):
                print(
                    f"Killing process {proc.info['pid']}: {' '.join(proc.info['cmdline'])}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


if __name__ == "__main__":

    # model_name = 'BAAI/bge-small-en-v1.5'
    # or
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    # or
    # model_name = 'intfloat/e5-small-v2'

    start_time = time.time()

    print(f"Starting to create embeddings for {model_name}")
    create_sharded_embeddings(
        '/workspace/trec-tot-2025-corpus.jsonl.zst',
        model_name,
        batch_size=512,
        shard_size=100000,
        output_dir='embeddings_shards',
        start_entry=100352
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    shard_files = list_shards('embeddings_shards')
    print(f"Found {len(shard_files)} shards")
    total_size = sum(shard['size_mb'] for shard in shard_files)
    print(f"Total size: {total_size:.1f} MB")
