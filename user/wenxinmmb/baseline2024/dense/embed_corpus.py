import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pyarrow as pa
import pyarrow.parquet as pq

# Usage:
# This script processes a JSONL dataset, computes embeddings using a specified SentenceTransformer model,
# and saves the embeddings in Parquet format. It supports resuming from the last processed line to avoid reprocessing.
# Example command to run the script:
# $ python embed_corpus.py --corpus_path $DATA_PATH/corpus.jsonl --model_name all-MiniLM-L6-v2 --output_folder ./embeddings/2025/miniLM

def get_last_processed_line(meta_path):
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return int(f.read().strip())
    return 0

def update_last_processed_line(meta_path, line_num):
    with open(meta_path, "w") as f:
        f.write(str(line_num))

def main(corpus_path, model_name, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    meta_path = os.path.join(output_folder, "progress.meta")
    last_line = get_last_processed_line(meta_path)
    batch_size = 1_000_000

    model = SentenceTransformer(model_name)
    embeddings = []
    doc_ids = []
    batch_idx = last_line // batch_size
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

            emb = model.encode(text)
            doc_ids.append(doc_id)
            embeddings.append(emb)
            current_line += 1

            if len(embeddings) == batch_size:
                df = pd.DataFrame({"id": doc_ids, "embedding": embeddings})
                table = pa.Table.from_pandas(df)
                pq.write_table(table, os.path.join(output_folder, f"embeddings_{batch_idx:05d}.parquet"))
                batch_idx += 1
                embeddings = []
                doc_ids = []
                update_last_processed_line(meta_path, current_line)

    # Write remaining
    if embeddings:
        df = pd.DataFrame({"id": doc_ids, "embedding": embeddings})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, os.path.join(output_folder, f"embeddings_{batch_idx:05d}.parquet"))
        update_last_processed_line(meta_path, current_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_folder", required=True)
    args = parser.parse_args()
    main(args.corpus_path, args.model_name, args.output_folder)