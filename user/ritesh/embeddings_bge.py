import os
import gc
import time
import torch
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def build_model(model_name="BAAI/bge-m3", device=None, use_fp16=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
    return model


def doc_to_chunks(text, tokenizer, max_len=512, stride=256, min_tokens=16):
    """
    Convert a long document into overlapping token windows (by token ids),
    then batch-decode to text chunks for embedding.
    """
    # Get token ids once (no truncation)
    ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    if len(ids) <= max_len:
        return [text]  # short doc -> single chunk

    chunks_ids = []
    step = max(1, max_len - stride)
    for start in range(0, len(ids), step):
        window = ids[start:start + max_len]
        if len(window) < min_tokens:
            break  # tiny tail
        chunks_ids.append(window)
        if start + max_len >= len(ids):
            break

    # Decode all windows at once (faster than per-window decode)
    chunks = tokenizer.batch_decode(chunks_ids, skip_special_tokens=True)
    return chunks if chunks else [text]


def encode_texts_dense(model, texts, batch_size=64, max_length=512, normalize=True, show_progress=True):
    """
    Encode a list of texts to dense vectors. Returns np.ndarray (N, 1024), float32.
    """
    out = model.encode(
        texts,
        return_dense=True, return_sparse=False, return_colbert_vecs=False,
        batch_size=batch_size,
        max_length=max_length,                 # this is the *chunk* window size
    )
    vecs = out["dense_vecs"].astype(np.float32)  # (N, 1024) float32, L2-normalized

    if normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
    return vecs


def embed_docs_with_chunking(model, titles, texts, batch_size=64, max_len=512, stride=256):
    """
    For each doc (title + text), create chunks by tokens, embed each chunk,
    then mean-pool chunk vectors to a single 1024-D vector per doc.
    Returns np.ndarray (N, 1024), float32.
    """
    tok = model.tokenizer
    # 1) Build chunk lists per doc
    doc_chunks = []
    doc_chunk_counts = []
    for title, body in zip(titles, texts):
        # simple concat; keep title to boost topical signal
        full = f"{title} {body}" if isinstance(title, str) else (body or "")
        chunks = doc_to_chunks(full, tok, max_len=max_len, stride=stride)
        doc_chunks.append(chunks)
        doc_chunk_counts.append(len(chunks))

    # 2) Flatten chunks and embed in batches
    all_chunks = [c for chunks in doc_chunks for c in chunks]
    if len(all_chunks) == 0:
        return np.zeros((len(texts), 1024), dtype=np.float32)

    chunk_vecs = encode_texts_dense(
        model, all_chunks, batch_size=batch_size, max_length=max_len, normalize=True, show_progress=True
    )  # (sum_chunks, 1024)

    # 3) Mean-pool back to per-doc vectors
    per_doc_vecs = np.zeros((len(texts), chunk_vecs.shape[1]), dtype=np.float32)
    idx = 0
    for i, count in enumerate(doc_chunk_counts):
        if count == 1:
            per_doc_vecs[i] = chunk_vecs[idx]
            idx += 1
        else:
            per_doc_vecs[i] = chunk_vecs[idx:idx + count].mean(axis=0)
            idx += count

    # sanity
    assert idx == len(all_chunks)
    return per_doc_vecs


def write_parquet_with_embeddings(out_path, id_vals, titles, urls, texts, embeddings_fp16):
    """
    Write a compact Parquet with a single list column for embeddings (float16).
    """
    assert embeddings_fp16.dtype == np.float16
    # Convert float16 arrays to Python lists for PyArrow
    emb_arrays = embeddings_fp16.astype(np.float32).tolist()
    
    table = pa.Table.from_pydict({
        "id": [str(x) for x in id_vals],
        "title": [x if isinstance(x, str) else "" for x in titles],
        "url": [x if isinstance(x, str) else "" for x in urls],
        "text": [x if isinstance(x, str) else "" for x in texts],
        "embedding": emb_arrays,
    })
    pq.write_table(table, out_path, compression="zstd")


def create_embeddings_from_parquet_shards(
    topic_name="entertainment",
    input_dir="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/cleaned_articles_parquet/",
    output_dir="/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/bge-m3-embeddings_shards_from_parquet_cleaned",
    model_name="BAAI/bge-m3",
    batch_size=1024,              # 512 was aggressive; 128–256 is safer with fp16
    max_len=512,                 # chunk window
    stride=256,                  # chunk overlap
    rows_per_batch=100000,        # process DF in slices to reduce RAM pressure
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name=model_name, device=device, use_fp16=True)

    print(f"Model: {model_name} | Device: {device} | Token limit reported: {model.tokenizer.model_max_length}")
    print(f"Chunking: max_len={max_len}, stride={stride}")

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".parquet"):
            continue
        if fname != f"{topic_name}_cleaned.parquet":
            continue

        input_path = os.path.join(input_dir, fname)
        print(f"\nProcessing {input_path}")
        start_time = time.time()

        # Load once (could also stream row groups via pyarrow.dataset if needed)
        df = pd.read_parquet(input_path)
        print(f"number of rows to process: {df.shape}")
        # Required columns
        id_col = "doc_id" if "doc_id" in df.columns else "id" if "id" in df.columns else None
        if id_col is None or "title" not in df.columns or "url" not in df.columns or "text" not in df.columns:
            print(f"Skipping {fname}: missing id/title/url/text cols.")
            continue

        model_short = model_name.split("/")[-1]
        out_fname = fname.replace(".parquet", f"_emb_{model_short}.parquet")
        out_path = os.path.join(output_dir, out_fname)

        # Process in row batches to keep memory in check
        N = len(df)
        write_mode = "truncate"
        for start in range(0, N, rows_per_batch):
            end = min(start + rows_per_batch, N)
            batch = df.iloc[start:end]

            titles = batch["title"].tolist()
            urls = batch["url"].tolist()
            texts = batch["text"].tolist()
            ids = batch[id_col].tolist()

            # Embed with chunking + mean-pool
            vecs = embed_docs_with_chunking(
                model, titles, texts, batch_size=batch_size, max_len=max_len, stride=stride
            )  # (B, 1024), float32, normalized

            # vecs_fp16 = vecs.astype(np.float32)

            # Write (append) to Parquet
            # We’ll write batch-by-batch: first batch creates file, later batches append
            # Convert float16 arrays to PyArrow arrays properly
            emb_arrays = []
            for v in vecs:
                # Convert to float32 first, then to PyArrow array
                v_list = v.astype(np.float32).tolist()
                emb_arrays.append(v_list)
            table = pa.Table.from_pydict({
                "id": [str(x) for x in ids],
                "title": [x if isinstance(x, str) else "" for x in titles],
                "url": [x if isinstance(x, str) else "" for x in urls],
                "text": [x if isinstance(x, str) else "" for x in texts],
                "embedding": emb_arrays,
            })
            if write_mode == "truncate":
                pq.write_table(table, out_path, compression="zstd")
                write_mode = "append"
            else:
                # append to existing file
                existing_table = pq.read_table(out_path)
                combined_table = pa.concat_tables([existing_table, table])
                pq.write_table(combined_table, out_path, compression="zstd")

            # cleanup
            del table, batch, ids, titles, urls, texts, vecs, emb_arrays
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # make sure the number of rows in the file is the same as the number of rows in the input file
        num_rows_in_output_file = pq.read_table(out_path).num_rows
        num_rows_in_input_file = len(df)

        end_time = time.time()
        print(f"Saved embeddings to {out_path}")
        print(f"Time taken for {fname}: {end_time - start_time:.1f} seconds")

    print("Done.")
    return num_rows_in_output_file, num_rows_in_input_file


# ---- topics list ----
topics = [
    "adult_content","art_design","crime_law","education_jobs","electronics_hardware",
    "entertainment","fashion_beauty","finance_business","food_dining","games","health",
    "history_geography","home_hobbies","industrial","literature","politics","religion",
    "science_math_technology","social_life","software","software_development",
    "sports_fitness","transportation","travel_tourism"
]

# Example: just one topic for now
output_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/bge-m3-embeddings_shards_from_parquet_cleaned"

topic_name = "finance_business"
num_rows_in_output_file, num_rows_in_input_file = create_embeddings_from_parquet_shards(topic_name=topic_name)
# Create a dummy file to confirm completion
done_path = os.path.join(output_dir, f"DONE_{topic_name}.txt")
with open(done_path, "w") as f:
    f.write(f"Embeddings for topic {topic_name} creation completed.\n")
    f.write(f"number of rows in the output file: {num_rows_in_output_file}\n")
    f.write(f"number of rows in the input file: {num_rows_in_input_file}\n")

topic_name = "food_dining"
num_rows_in_output_file, num_rows_in_input_file = create_embeddings_from_parquet_shards(topic_name=topic_name)
# Create a dummy file to confirm completion
done_path = os.path.join(output_dir, f"DONE_{topic_name}.txt")
with open(done_path, "w") as f:
    f.write(f"Embeddings for topic {topic_name} creation completed.\n")
    f.write(f"number of rows in the output file: {num_rows_in_output_file}\n")
    f.write(f"number of rows in the input file: {num_rows_in_input_file}\n")

topic_name = "games"
num_rows_in_output_file, num_rows_in_input_file = create_embeddings_from_parquet_shards(topic_name=topic_name)
# Create a dummy file to confirm completion
done_path = os.path.join(output_dir, f"DONE_{topic_name}.txt")
with open(done_path, "w") as f:
    f.write(f"Embeddings for topic {topic_name} creation completed.\n")
    f.write(f"number of rows in the output file: {num_rows_in_output_file}\n")
    f.write(f"number of rows in the input file: {num_rows_in_input_file}\n")

topic_name = "health"
num_rows_in_output_file, num_rows_in_input_file = create_embeddings_from_parquet_shards(topic_name=topic_name)
# Create a dummy file to confirm completion
done_path = os.path.join(output_dir, f"DONE_{topic_name}.txt")
with open(done_path, "w") as f:
    f.write(f"Embeddings for topic {topic_name} creation completed.\n")
    f.write(f"number of rows in the output file: {num_rows_in_output_file}\n")
    f.write(f"number of rows in the input file: {num_rows_in_input_file}\n")

topic_name = "home_hobbies"
num_rows_in_output_file, num_rows_in_input_file = create_embeddings_from_parquet_shards(topic_name=topic_name)
# Create a dummy file to confirm completion
done_path = os.path.join(output_dir, f"DONE_{topic_name}.txt")
with open(done_path, "w") as f:
    f.write(f"Embeddings for topic {topic_name} creation completed.\n")
    f.write(f"number of rows in the output file: {num_rows_in_output_file}\n")
    f.write(f"number of rows in the input file: {num_rows_in_input_file}\n")

topic_name = "industrial"
num_rows_in_output_file, num_rows_in_input_file = create_embeddings_from_parquet_shards(topic_name=topic_name)
# Create a dummy file to confirm completion
done_path = os.path.join(output_dir, f"DONE_{topic_name}.txt")
with open(done_path, "w") as f:
    f.write(f"Embeddings for topic {topic_name} creation completed.\n")
    f.write(f"number of rows in the output file: {num_rows_in_output_file}\n")
    f.write(f"number of rows in the input file: {num_rows_in_input_file}\n")

topic_name = "literature"
num_rows_in_output_file, num_rows_in_input_file = create_embeddings_from_parquet_shards(topic_name=topic_name)
# Create a dummy file to confirm completion
done_path = os.path.join(output_dir, f"DONE_{topic_name}.txt")
with open(done_path, "w") as f:
    f.write(f"Embeddings for topic {topic_name} creation completed.\n")
    f.write(f"number of rows in the output file: {num_rows_in_output_file}\n")
    f.write(f"number of rows in the input file: {num_rows_in_input_file}\n")