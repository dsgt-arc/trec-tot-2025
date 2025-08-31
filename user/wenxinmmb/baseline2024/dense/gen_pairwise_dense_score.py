import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import pyarrow.parquet as pq
from datetime import datetime

DATA_PATH = os.environ.get("DATA_PATH", "/path/to/data")
QUERY_EMB_PATH = os.path.join(DATA_PATH, "2025/all_query_embedding_combined.parquet")
DOC_EMB_DIR = os.path.join(DATA_PATH, "upstash-embeddings-condensed/data")
OUTPUT_TSV = "outputs/doc_to_query/all-sets.tsv"
DENSE_SCORE_TSV = "outputs/dense_score/all-sets.tsv"
MISSING_DOC_TSV = "outputs/missing_doc/all-sets.tsv"

# Load query embeddings into dict
query_df = pd.read_parquet(QUERY_EMB_PATH)
query_emb_dict = {str(row['query_id']): np.array(row['embedding']) for _, row in query_df.iterrows()}

doc_emb_files = sorted(glob(os.path.join(DOC_EMB_DIR, "*.parquet")))
print(doc_emb_files)

def output_tsv_iter(path):
    with open(path, 'r') as fin:
        group_doc_id = None
        queries = []
        for line in fin:
            line_doc_id, query_id = line.strip().split('\t')
            if group_doc_id is None:
                group_doc_id = line_doc_id
            if line_doc_id != group_doc_id:
                yield group_doc_id, queries
                group_doc_id = line_doc_id
                queries = []
            queries.append(query_id)
        if group_doc_id is not None:
            yield group_doc_id, queries

def write_dense_scores(doc_id, para_embs, queries, query_emb_dict, fout):
    global actual_score_count
    # print('write-dense-score for', doc_id)
    if not queries or not para_embs:
        for query_id in queries:
            fout.write(f"{doc_id}\t{query_id}\t-100.0\n")
        return
    para_embs_np = np.stack(para_embs)
    for query_id in queries:
        query_emb = query_emb_dict.get(query_id)
        if query_emb is not None:
            scores = np.dot(para_embs_np, query_emb)
            score = float(np.max(scores)) if scores.size > 0 else -100.0
            actual_score_count += 1
        else:
            score = -100.0
        fout.write(f"{doc_id}\t{query_id}\t{score}\n")

actual_score_count = 0
missing_docid_count = 0

with open(DENSE_SCORE_TSV, 'w') as fout, open(MISSING_DOC_TSV, 'w') as miss_fout:
    emb_doc_id = None
    current_emb_doc_id = None
    output_iter = output_tsv_iter(OUTPUT_TSV)
    try:
        tsv_doc_id, tsv_queries = next(output_iter)
    except StopIteration:
        tsv_doc_id, tsv_queries = None, []

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | First TSV doc_id: {tsv_doc_id}")
    print(f"First embedding doc_id: {current_emb_doc_id}")

    for idx, file in enumerate(tqdm(doc_emb_files, desc="Processing doc embeddings")):
        if idx % 5 == 0 and idx > 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Processed {idx} files")
        parquet_file = pq.ParquetFile(file)
        para_embs = []
        current_emb_doc_id = None

        for batch in parquet_file.iter_batches():
            batch_df = batch.to_pandas()
            for _, row in batch_df.iterrows():
                full_id = str(row['id'])
                emb_doc_id = int(full_id.split('_')[0])  # Convert to int
                emb = np.array(row['embedding'])

                # Initialize current_emb_doc_id
                if current_emb_doc_id is None:
                    current_emb_doc_id = emb_doc_id
                # if current_emb_doc_id == 308:
                    # print('Processing doc_id 308', tsv_doc_id, emb_doc_id, current_emb_doc_id)

                # If we've moved to a new doc_id in embeddings
                if emb_doc_id != current_emb_doc_id:
                    # Synchronize TSV iterator: advance if TSV doc_id is behind
                    while tsv_doc_id is not None and int(tsv_doc_id) < int(current_emb_doc_id):
                        missing_docid_count += 1
                        miss_fout.write(f"{tsv_doc_id}\n")  # Write missing doc_id
                        try:
                            tsv_doc_id, tsv_queries = next(output_iter)
                            tsv_doc_id = int(tsv_doc_id)
                        except StopIteration:
                            tsv_doc_id, tsv_queries = None, []

                    # If doc_id matches, score it
                    try:
                        if int(tsv_doc_id) == int(current_emb_doc_id):
                            # if int(tsv_doc_id) == 308:
                                # print('Scoring doc_id 308 (L104)', tsv_doc_id, emb_doc_id, current_emb_doc_id)
                            write_dense_scores(current_emb_doc_id, para_embs, tsv_queries, query_emb_dict, fout)
                            try:
                                tsv_doc_id, tsv_queries = next(output_iter)
                                tsv_doc_id = int(tsv_doc_id)  # Ensure integer
                            except StopIteration:
                                tsv_doc_id, tsv_queries = None, []
                    except Exception as e:
                        print(f"Error scoring doc_id {current_emb_doc_id}, {tsv_doc_id}: {e}")
                        raise e

                    # Reset for next doc_id
                    para_embs = []
                    current_emb_doc_id = emb_doc_id

                # if tsv_doc_id == current_emb_doc_id:
                    # para_embs.append(emb)
                para_embs.append(emb)

        # Handle last doc_id in file
        if current_emb_doc_id is not None:
            while tsv_doc_id is not None and int(tsv_doc_id) < current_emb_doc_id:
                missing_docid_count += 1
                miss_fout.write(f"{tsv_doc_id}\n")  # Write missing doc_id
                try:
                    tsv_doc_id, tsv_queries = next(output_iter)
                    tsv_doc_id = int(tsv_doc_id)
                except StopIteration:
                    tsv_doc_id, tsv_queries = None, []
            if tsv_doc_id == current_emb_doc_id:
                write_dense_scores(current_emb_doc_id, para_embs, tsv_queries, query_emb_dict, fout)
                try:
                    tsv_doc_id, tsv_queries = next(output_iter)
                    tsv_doc_id = int(tsv_doc_id)  # Ensure integer
                except StopIteration:
                    tsv_doc_id, tsv_queries = None, []

print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Number of actual scores computed: {actual_score_count}")
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Number of missing doc_ids in embeddings: {missing_docid_count}")