import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pyarrow as pa
import pyarrow.parquet as pq
import datetime

# Usage:
# This script processes a JSONL dataset, computes embeddings using a specified SentenceTransformer model,
# and saves the embeddings in Parquet format. It supports resuming from the last processed line to avoid reprocessing.
# Example command to run the script:
# $ python embed_corpus.py --corpus_path $DATA_PATH/corpus.jsonl --model_name all-MiniLM-L6-v2 --output_folder ./embeddings/2025/MiniLM

def get_last_processed_line(meta_path):
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return int(f.read().strip())
    return 0

def update_last_processed_line(meta_path, line_num):
    with open(meta_path, "w") as f:
        f.write(str(line_num))

def main(corpus_path, model_name, output_folder, parquet_batch_size, encode_batch_size):
    os.makedirs(output_folder, exist_ok=True)
    meta_path = os.path.join(output_folder, "progress.meta")
    info_path = os.path.join(output_folder, "embedding_info.txt")
    last_line = get_last_processed_line(meta_path)

    model = SentenceTransformer(model_name)
    doc_ids = []
    texts = []
    all_ids = []
    all_embeddings = []
    batch_idx = last_line // parquet_batch_size
    current_line = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing", initial=last_line):
            if current_line < last_line:
                current_line += 1
                continue

            obj = json.loads(line)
            doc_id = obj.get("id")
            text = obj.get("text")
            if text is None or doc_id is None:
                current_line += 1
                continue

            doc_ids.append(doc_id)
            texts.append(text)
            current_line += 1

            # When enough texts for encode batch, encode and accumulate
            if len(texts) == encode_batch_size:
                embeddings = model.encode(texts, batch_size=encode_batch_size, show_progress_bar=False)
                all_ids.extend(doc_ids)
                all_embeddings.extend(embeddings)
                doc_ids = []
                texts = []

            # When enough embeddings for parquet batch, write to parquet
            if len(all_embeddings) >= parquet_batch_size:
                df = pd.DataFrame({"id": all_ids[:parquet_batch_size], "embedding": list(all_embeddings[:parquet_batch_size])})
                table = pa.Table.from_pandas(df)
                pq.write_table(table, os.path.join(output_folder, f"embeddings_{batch_idx:05d}.parquet"))
                batch_idx += 1
                all_ids = all_ids[parquet_batch_size:]
                all_embeddings = all_embeddings[parquet_batch_size:]
                update_last_processed_line(meta_path, current_line)

    # Encode any remaining texts
    if texts:
        embeddings = model.encode(texts, batch_size=encode_batch_size, show_progress_bar=False)
        all_ids.extend(doc_ids)
        all_embeddings.extend(embeddings)

    # Write any remaining embeddings to parquet
    if all_embeddings:
        df = pd.DataFrame({"id": all_ids, "embedding": list(all_embeddings)})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, os.path.join(output_folder, f"{batch_idx:05d}.parquet"))
        update_last_processed_line(meta_path, current_line)
    
    # Write model name and date to info file
    with open(info_path, "w") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"date: {datetime.datetime.now().isoformat()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--parquet_batch_size", type=int, default=1_000_000, help="Number of embeddings per parquet file")
    parser.add_argument("--encode_batch_size", type=int, default=32, help="Batch size for model.encode")
    args = parser.parse_args()
    # ensure parquet batch size is a multiple of encode batch size
    assert args.parquet_batch_size % args.encode_batch_size == 0, "parquet_batch_size must be a multiple of encode_batch_size"

    main(args.corpus_path, args.model_name, args.output_folder, args.parquet_batch_size, args.encode_batch_size)