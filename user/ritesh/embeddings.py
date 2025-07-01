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


def create_sharded_embeddings(jsonl_path,
                              model_name,
                              batch_size=512,
                              max_entries=None,
                              output_prefix='wikipedia',
                              shard_size=1000000,
                              output_dir='embeddings_shards'):
    """
    Create embeddings and save as multiple smaller Parquet files
    """

    print(f"Loading model: {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    dimension = model.get_sentence_embedding_dimension()
    print(f"Model dimension: {dimension}")

    # Create output directory
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

        for line in tqdm(f, desc="Processing documents"):

            # Check if we've reached the limit
            if max_entries is not None and processed_count >= max_entries:
                break

            data = json.loads(line.strip())

            # Format text for E5/BGE models
            # E5 models expect "query: " or "passage: " prefix
            # BGE models work well with just the text
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
                # Generate embeddings
                embeddings = model.encode(
                    batch_texts, show_progress_bar=False, batch_size=batch_size)

                # Combine metadata and embeddings
                for j, (emb, meta) in enumerate(zip(embeddings, batch_metadata)):
                    row = meta.copy()
                    # Add embedding columns
                    row.update({f'emb_{k}': float(v)
                               for k, v in enumerate(emb)})
                    all_data.append(row)

                processed_count += len(batch_texts)

                # Save shard when we reach shard_size
                if len(all_data) >= shard_size:
                    shard_count += 1
                    save_shard(all_data, shard_count, model_name, output_dir)
                    all_data = []  # Clear memory

                # print progress
                if processed_count % batch_size == 0:
                    elapsed_time = time.time() - start_time
                    docs_per_second = processed_count / elapsed_time
                    print(f"Processed {processed_count:,} documents | "
                          f"Rate: {docs_per_second:.1f} docs/s | "
                          f"Elapsed: {elapsed_time/60:.1f} minutes")

                # Clear batch
                batch_texts = []
                batch_metadata = []

        # Process remaining items
        if batch_texts and (max_entries is None or processed_count < max_entries):
            remaining_slots = max_entries - \
                processed_count if max_entries else len(batch_texts)
            if max_entries:
                batch_texts = batch_texts[:remaining_slots]
                batch_metadata = batch_metadata[:remaining_slots]

            embeddings = model.encode(
                batch_texts, show_progress_bar=False, batch_size=batch_size)

            for j, (emb, meta) in enumerate(zip(embeddings, batch_metadata)):
                row = meta.copy()
                row.update({f'emb_{k}': float(v) for k, v in enumerate(emb)})
                all_data.append(row)

            processed_count += len(batch_texts)

    # Save final shard if there's remaining data
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
    filename = f"shard_{shard_number:04d}_{model_short}_{len(data):,}docs_{timestamp}.parquet"
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


if __name__ == "__main__":

    model_name = 'BAAI/bge-base-en-v1.5'

    start_time = time.time()

    print(f"Starting to create embeddings for {model_name}")
    create_sharded_embeddings(
        '/workspace/trec-tot-2025-corpus.jsonl.zst',
        model_name,
        batch_size=512,
        shard_size=100,
        output_dir='embeddings_shards'
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    shard_files = list_shards('embeddings_shards')
    print(f"Found {len(shard_files)} shards")
    total_size = sum(shard['size_mb'] for shard in shard_files)
    print(f"Total size: {total_size:.1f} MB")
