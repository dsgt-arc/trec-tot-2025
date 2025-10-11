import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
import time
import os
from datetime import datetime
import torch
from collections import defaultdict
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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


def get_full_text_embeddings_batched(docs, model, max_tokens=256, batch_size=512, stride=512, device='cuda'):
    tokenizer = model.tokenizer

    # Batch tokenize with overflow
    tokenized = tokenizer(
        docs,
        max_length=max_tokens,
        stride=stride,
        truncation=True,
        return_overflowing_tokens=True,
        return_attention_mask=False,
        return_tensors=None,
        padding=False
    )
    # Decode tokens to string chunks
    chunked_texts = tokenizer.batch_decode(tokenized['input_ids'], skip_special_tokens=True)
    
    # Track which doc each chunk belongs to
    doc_indices = tokenized['overflow_to_sample_mapping']

    # Batch encode all chunks
    embeddings = model.encode(
        chunked_texts,
        show_progress_bar=True,
        batch_size=batch_size,
        use_fp16=True,
        device=device,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # chunked_texts = []
    # doc_indices = []  # Track which doc each chunk belongs to
    # for idx, text in enumerate(docs):
    #     chunks = chunk_text(text, tokenizer, max_tokens=max_tokens, stride=stride)
    #     chunked_texts.extend(chunks)
    #     doc_indices.extend([idx] * len(chunks))
    # Batch encode all chunks
    # embeddings = model.encode(chunked_texts, show_progress_bar=False, batch_size=batch_size)
    # Group by doc and mean-pool
    # doc_to_embs = defaultdict(list)
    # for emb, doc_idx in zip(embeddings, doc_indices):
    #     doc_to_embs[doc_idx].append(emb)
    # pooled = [np.mean(doc_to_embs[i], axis=0) for i in range(len(docs))]

    # Pre-allocate a structure to hold the sum of embeddings and count
    embedding_dim = len(embeddings[0])
    sums = np.zeros((len(docs), embedding_dim), dtype=np.float32)
    counts = np.zeros(len(docs), dtype=np.int32)

    # Accumulate
    for emb, doc_idx in zip(embeddings, doc_indices):
        sums[doc_idx] += emb
        counts[doc_idx] += 1

    # Avoid division by zero (in case any doc had no chunks)
    counts = np.maximum(counts, 1)
    pooled = sums / counts[:, None]

    return pooled

def create_embeddings_from_parquet_shards(
    input_dir="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/split_parquet_shards/",
    output_dir="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/bge-m3-embeddings_shards_from_parquet",
    model_name="BAAI/bge-m3",
    batch_size=512,
):
    """
    Loop through each Parquet file in input_dir, compute embeddings, and save to output_dir.
    Only keep doc_id (or id), title, url, text, and embedding columns in the output.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    max_tokens = model.tokenizer.model_max_length
    stride = max_tokens // 2
    print(f"Model: {model_name} | Max tokens: {max_tokens}")

    for fname in sorted(os.listdir(input_dir)):
        start_time = time.time()
        if not fname.endswith('.parquet'):
            continue
        input_path = os.path.join(input_dir, fname)
        print(f"Processing {input_path}")
        df = pd.read_parquet(input_path)
        # Use 'doc_id' if present, else 'id'
        id_col = 'doc_id' if 'doc_id' in df.columns else 'id' if 'id' in df.columns else None
        if id_col is None or 'title' not in df.columns or 'url' not in df.columns or 'text' not in df.columns:
            print(f"Skipping {fname}: missing one of 'doc_id'/'id', 'title', 'url', or 'text' columns.")
            continue
        # Prepare texts
        if 'e5' in model_name.lower():
            texts = [f"passage: {row['title']} {row['text']}" for _, row in df.iterrows()]
        else:
            texts = [f"{row['title']} {row['text']}" for _, row in df.iterrows()]
        # Compute embeddings in batches
        pooled_embeddings = get_full_text_embeddings_batched(texts, model, max_tokens=max_tokens, batch_size=batch_size, stride=stride)
        # Build new DataFrame with only required columns and embeddings
        out_df = df[[id_col, 'title', 'url', 'text']].copy()
        emb_dim = len(pooled_embeddings[0])
        for k in range(emb_dim):
            out_df[f'emb_{k}'] = [float(emb[k]) for emb in pooled_embeddings]
        # Save to output_dir
        model_short = model_name.split("/")[-1]
        out_fname = fname.replace('.parquet', f'_emb_{model_short}.parquet')
        out_path = os.path.join(output_dir, out_fname)
        out_df.to_parquet(out_path, index=False, compression="zstd")
        print(f"Saved embeddings to {out_path}")
        # Clear DataFrames and free memory
        torch.cuda.empty_cache()
        del out_df
        del df
        del pooled_embeddings
        del texts
        gc.collect()
        end_time = time.time()
        print(f"Time taken for {fname}: {end_time - start_time} seconds")
    return


if __name__ == "__main__":

    # model_name = 'BAAI/bge-small-en-v1.5'
    # or
    # model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    # or
    model_name = 'BAAI/bge-m3'

    start_time = time.time()

    # Uncomment below to run on Parquet shards
    create_embeddings_from_parquet_shards(
        input_dir="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/split_parquet_shards/",
        output_dir="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/bge-m3-embeddings_shards_from_parquet",
        model_name=model_name,
        batch_size=16
    )

    end_time = time.time()
    print(f"Time taken for all shards: {end_time - start_time} seconds")

