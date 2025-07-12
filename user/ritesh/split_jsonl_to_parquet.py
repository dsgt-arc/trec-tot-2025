import os
import json
import pandas as pd
import zstandard as zstd
from tqdm import tqdm

def split_jsonl_zst_to_parquet(jsonl_zst_path, output_dir, shard_size=100_000):
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the zstd-compressed file
    with zstd.open(jsonl_zst_path, 'rt', encoding='utf-8') as f:
        batch = []
        shard_count = 0
        for i, line in enumerate(tqdm(f, desc="Reading and splitting")):
            data = json.loads(line.strip())
            batch.append(data)
            if len(batch) >= shard_size:
                shard_count += 1
                save_parquet_shard(batch, shard_count, output_dir)
                batch = []
        # Save any remaining data
        if batch:
            shard_count += 1
            save_parquet_shard(batch, shard_count, output_dir)
    print(f"Done. Created {shard_count} shards in {output_dir}")

def save_parquet_shard(data, shard_number, output_dir):
    df = pd.DataFrame(data)
    filename = f"shard_{shard_number:04d}_{len(df)}docs.parquet"
    filepath = os.path.join(output_dir, filename)
    df.to_parquet(filepath, index=False, compression="zstd")
    print(f"Saved {filename} with {len(df)} entries.")

if __name__ == "__main__":
    jsonl_zst_path = "/workspace/trec-tot-2025-corpus.jsonl.zst"
    output_dir = "split_parquet_shards"
    split_jsonl_zst_to_parquet(jsonl_zst_path, output_dir, shard_size=100_000)
