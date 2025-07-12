import argparse
import json
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import ir_datasets
import os
import polars as pl
import glob
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils

log = logging.getLogger(__name__)

# Run commands: python dense/dense_search_generic.py --data_version 2025 --data_path $DATA_PATH

def do_partition_search(fl_pattern, res_name, split_name, model_name, device, top_k):

    # Read all files into a single DataFrame
    directory = args.embed_src_dir
    file_pattern = f"{directory}/{fl_pattern}"
    file_paths = glob.glob(file_pattern)
    df = pl.read_parquet(file_paths)

    # Initialize FAISS index
    embedding_size = len(df['embedding'].iloc[0])
    log.info("Embedding size:", embedding_size)
    index = faiss.IndexFlatIP(embedding_size) # IndexFlatIP uses inner product (dot product) for similarity

    embeddings = np.vstack(df['embedding'].to_numpy())
    index.add(embeddings)
    log.info(f"Added {df.shape[0]} records to FAISS index")

    # load encoder and calculate query embeddings
    transformer = SentenceTransformer(
        model_name,
        device=device
    )

    encode_batch_size = 8
    queries = []
    dataset = ir_datasets.load(split_name)
    
    all_scores = np.empty((0, top_k))
    all_raw_doc_ids = np.empty((0, top_k), dtype=int)
    qids = []
    
    def _calculate_score_for_batch(query_batch):
        query_vectors = transformer.encode(
            sentences=query_batch,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        scores, raw_doc_ids = index.search(query_vectors, k=top_k)
        log.info(f"Score: \n{scores}, \nDoc ID: \n{raw_doc_ids}")
        return scores, raw_doc_ids

    for q in dataset.queries_iter():
        log.info('Processing query - %s: %s', q.query_id, q.query[:150])
        queries.append(q.query)
        qids.append(q.query_id)
        if len(queries) >= encode_batch_size:
            scores, raw_doc_ids = _calculate_score_for_batch(queries)
            all_scores = np.vstack((all_scores, scores))
            all_raw_doc_ids = np.vstack((all_raw_doc_ids, raw_doc_ids))
            queries = []

    if len(queries) > 0:
        scores, raw_doc_ids = _calculate_score_for_batch(queries)
        all_scores = np.vstack((all_scores, scores))
        all_raw_doc_ids = np.vstack((all_raw_doc_ids, raw_doc_ids))

    log.info(f"Total queries processed: scores shape -- {all_scores.shape}, "
                f"raw_doc_ids shape -- {all_raw_doc_ids.shape}")
    
    def get_doc_id(id, df):
        # assert id >= 0, "Doc ID should be non-negative"
        df_entry = df.slice(id, 1).select(["id", "title", "url"]).to_dicts()[0]
        # log.info(f"search result id {id} --> doc_id {df_entry['id']}, title {df_entry['title']}")
        return str(df_entry['id'])

    # Translate raw_doc_ids to df_entry['id'] as strings
    translated_ids = [
        [get_doc_id(id, df) for id in id_rows]
        for id_rows in all_raw_doc_ids
    ]

    # write the results to a file
    with open(res_name, 'w') as f:
        for qid, sc, rdoc_ids, wikip_ids in zip(qids, all_scores, all_raw_doc_ids, translated_ids):
            result = {
                "query_id": qid,
                "scores": sc.tolist(),
                "raw_doc_ids": rdoc_ids.tolist(),
                "translated_doc_ids": wikip_ids,
            }
            json.dump(result, f)
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("dense_search_generic",
                        description="semantic search on pre-computed embeddings found on huggingface")
    parser.add_argument("--data_path", required=True, help="location to ToT dataset")
    parser.add_argument("--data_version", default="2025",
                        help="data version to use, e.g., 2024 or 2025")
    parser.add_argument("--embed_src_dir", required=True,
                        help="location to the folder that stores the embedding parquet files")
    parser.add_argument("--top_k", type=int, default=1000,
                        help="number of top results to return for each query")
    parser.add_argument("--model_name", required=True,
                        help="name of the SentenceTransformer model to use for encoding queries, i.e BAAI/bge-m3")
    parser.add_argument("--num_files_per_search", type=int, default=1,
                        help="number of files to process in each search partition, e.g., 1, 2, 3, etc.")

    # setting log levels
    logging.basicConfig(level=logging.INFO)
    log.setLevel(logging.INFO)

    # FIXME: set OMP thread to 1, otherwise FAISS CPU search hit memory leak on mac os
    # os.environ["OMP_NUM_THREADS"] = "1"

    # auto-detect device 
    device = utils.get_device()

    args = parser.parse_args()
    
    if args.data_version == "2025":
        import tot_25 as tot
    elif args.data_version == "2024":
        import tot_24 as tot
    else:
        raise ValueError(f"Unknown data version: {args.data_version}")
    tot.register(args.data_path)

    split_name = 'trec-tot:train-2025'

    dir_prefix = 'runs/search_result/train/results_'
    for i in range (0, 5):
        do_partition_search(f'{i}[0-1][0-9].parquet', f'{dir_prefix}000_009.json', split_name, device)
        do_partition_search(f'{i}[2-5][0-9].parquet', f'{dir_prefix}020_059.json', split_name, device)
        do_partition_search(f'{i}[6-9][0-9].parquet', f'{dir_prefix}060_099.json', split_name, device)

