import pandas as pd
import json
import os
import glob
import datetime

DATA_PATH = os.environ.get("DATA_PATH", ".")
CORPUS_MAPPING_PATH = os.path.join(DATA_PATH, "2025", "corpus-offset-mapping.json")
INPUT_DIR = "/home/wenxin/project/data/upstash-embeddings/data/en/"
OUTPUT_PARQUET_DIR = "/home/wenxin/project/data/upstash-embeddings-condensed/data"
OUTPUT_CSV_DIR = "/home/wenxin/project/data/upstash-embeddings-condensed/removed_csv"

os.makedirs(OUTPUT_PARQUET_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# Load corpus-offset-mapping.json
with open(CORPUS_MAPPING_PATH, "r") as f:
    corpus_mapping = json.load(f)
valid_article_ids = set(corpus_mapping.keys())

# Helper to get article_id from id
def get_article_id(paragraph_id):
    return paragraph_id.split("_")[0]

# Condense text to first 5 words
def first_5_words(text):
    return " ".join(text.split()[:5])

parquet_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))

LOG_FILE = os.path.join(OUTPUT_PARQUET_DIR, "process.log")
for i in range(0, len(parquet_files), 10):
    dfs = []
    removed_dfs = []
    group_files = parquet_files[i:i+10]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_names = [os.path.basename(f) for f in group_files]
    log_msg = f'[{timestamp}] Processing gf: {file_names}'
    print(log_msg)
    with open(LOG_FILE, "a") as logf:
        logf.write(log_msg + "\n")
    for pf in group_files:
        df = pd.read_parquet(pf)
        keep_mask = df["id"].apply(lambda x: get_article_id(x) in valid_article_ids)
        removed = df[~keep_mask][["id", "url"]]
        removed["article_id"] = removed["id"].apply(get_article_id)
        removed_dfs.append(removed)
        df = df[keep_mask].copy()
        df["text"] = df["text"].apply(first_5_words)
        df = df[["id", "text", "embedding"]]
        dfs.append(df)
    out_df = pd.concat(dfs, ignore_index=True)
    out_removed = pd.concat(removed_dfs, ignore_index=True)
    # Consolidate by article_id
    consolidated = out_removed.groupby("article_id").first().reset_index()[["article_id", "url"]]
    consolidated.rename(columns={"article_id": "id"}, inplace=True)
    out_parquet_path = os.path.join(OUTPUT_PARQUET_DIR, f"{i//10:03d}.parquet")
    out_csv_path = os.path.join(OUTPUT_CSV_DIR, f"removed_{i//10:03d}.csv")
    out_df.to_parquet(out_parquet_path, index=False)
    consolidated.to_csv(out_csv_path, index=False)
    # Delete processed parquet files from original directory
    for pf in group_files:
        os.remove(pf)
