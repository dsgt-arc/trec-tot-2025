from datasets import load_dataset
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

log = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("dense_search", description="semantic search on pre-computed embeddings found on huggingface")
    parser.add_argument("--hf_dataset", default="Upstash/wikipedia-2024-06-bge-m3", help="the huggingface embedding dataset to use")
    parser.add_argument("--data_version", default="2025",
                        help="data version to use, e.g., 2024 or 2025")
    parser.add_argument("--data_path", default="./datasets/TREC-ToT2024/", help="location to dataset")
    parser.add_argument("--embed_src_dir", default="/Users/wenxin/tot/tot_data/upstash-embed/data/en/",
                        help="location that stores the embedding parquet files")
    # setting log levels
    logging.basicConfig(level=logging.INFO)
    log.setLevel(logging.INFO)

    os.environ["OMP_NUM_THREADS"] = "1" # FIXME: set OMP thread to 1, otherwise FAISS CPU search hit memory leak on mac os

    # auto-detect device 
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info(f"Using device: {device}")

    args = parser.parse_args()
    if args.data_version == "2025":
        import tot_25 as tot
    elif args.data_version == "2024":
        import tot_24 as tot
    else:
        raise ValueError(f"Unknown data version: {args.data_version}")
    tot.register(args.data_path)

    chunk_size = 100000 # number of embeddings to process at a time
    total_records = chunk_size*10  # max = 47_018_430
    chunk_data = []
    chunk_embeddings = []
    chunk_index = 0

    # Initialize FAISS index
    embedding_size = 1024
    index = faiss.IndexFlatIP(embedding_size) # this uses inner product (dot product) for similarity
    indexwmap = faiss.IndexIDMap(index)

    # There are 470 parquet files in total. There are 100K records per file, expect for the last file
    # When do we hit RAM limit in mac with 32GB RAM? paused at 64, 128 and start using swap storage at 64th. 
    cursor = 0
    chunk_file_number = 8  # 32
    max_file_number = 471  # total number of parquet files

    # TODO: add another for-loop to iterate over all files
    # for num in range(0, chunk_file_number):
    #     df = pl.read_parquet(f'{args.embed_src_dir}/{num:03d}.parquet')
    #     # print df stats
    #     log.info(f"Loaded dataset index {num} with {df.shape[0]} records.")
    #     # df.columns is ['id', 'url', 'title', 'text', 'embedding']
    #     # log.info(f"First record:\n{df.head(1)}")
    #     # adding df to indexwmap
    #     ids = np.arange(cursor, cursor + df.shape[0], dtype=np.int64)
    #     embeddings = np.vstack(df['embedding'].to_numpy())
    #     indexwmap.add_with_ids(embeddings, ids)
    #     cursor += df.shape[0]
    #     log.info(f"Added {df.shape[0]} records to FAISS index, total records in index: {cursor}")

    # Read all files into a single DataFrame
    directory = args.embed_src_dir
    # file_pattern = f"{directory}/0[0-3][0-9].parquet"  # Matches 000.parquet to 032.parquet
    file_pattern = f"{directory}/00[0-9].parquet"  # Matches 000.parquet to 032.parquet
    file_paths = glob.glob(file_pattern)
    df = pl.read_parquet(file_paths)
    # ids = np.arange(0, df.shape[0], dtype=np.int64)
    # setup ids as the df['id'] column. while ids are string, process it so that the string is split by _ and use the number before "_"
    # ids = df['id'].apply(lambda x: int(x.split('_')[0])).to_numpy(dtype=np.int64)
    df = df.with_columns(
        pl.col('id').str.split('_').list.get(0).cast(pl.Int64).alias('numeric_id')
    )
    ids = df['numeric_id'].to_numpy()

    embeddings = np.vstack(df['embedding'].to_numpy())
    indexwmap.add_with_ids(embeddings, ids)
    log.info(f"Added {df.shape[0]} records to FAISS index")

    # load encoder and calculate query embeddings
    transformer = SentenceTransformer(
        "BAAI/bge-m3",
        device=device,
        revision="babcf60cae0a1f438d7ade582983d4ba462303c2",
    )

    encode_batch_size = 8
    queries = []
    dataset = ir_datasets.load("trec-tot:train-2024")
    
    top_k = 10  # number of top results to retrieve
    all_scores = np.empty((0, top_k))
    all_raw_doc_ids = np.empty((0, top_k), dtype=int)
    for q in dataset.queries_iter():
        # log first a few characters of the query
        log.info('Processing query: %s %s', q.query_id, q.query[:150])
        queries.append(q.query)
        if len(queries) >= encode_batch_size:
            query_vectors = transformer.encode(
                sentences=queries,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            scores, raw_doc_ids = index.search(query_vectors, k=10)
            log.info(f"Score: \n{scores}, Doc ID: \n{raw_doc_ids}")
            # Aggregate scores and raw_doc_ids
            all_scores = np.vstack((all_scores, scores))
            all_raw_doc_ids = np.vstack((all_raw_doc_ids, raw_doc_ids))
            queries = []  # reset queries after processing batch

    if len(queries) > 0:
        # TODO: refactoring
        query_vectors = transformer.encode(
            sentences=queries,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        scores, raw_doc_ids = index.search(query_vectors, k=10)
        log.info(f"Score: \n{scores}, Doc ID: \n{raw_doc_ids}")
        # Aggregate scores and raw_doc_ids
        all_scores = np.vstack((all_scores, scores))
        all_raw_doc_ids = np.vstack((all_raw_doc_ids, raw_doc_ids))

    # print the shape of the all_scores and raw_doc_ids
    log.info(f"Total queries processed: scores shape -- {all_scores.shape}, raw_doc_ids shape -- {all_raw_doc_ids.shape}")
    # write the results to a file
    with open('search_results.json', 'a') as f:
        for qid, sc, rdoc_ids in zip([q.query_id]*len(scores), scores, raw_doc_ids):
            result = {
                "query_id": qid,
                "scores": sc.tolist(),
                "doc_ids": rdoc_ids.tolist()
            }
            json.dump(result, f)
            f.write('\n')
            
    print(raw_doc_ids.shape)
    


    def get_doc_id(id, df):
        assert id >= 0, "Doc ID should be non-negative"
        # file_number = id // 100000  # TODO: do not hardcode 100K
        # record_index = id % 100000
        # df_sub = pl.read_parquet(f'{embed_src_dir}/{file_number:03d}.parquet')
        # df_entry = df_sub.slice(record_index, 1).select(["id", "title", "url"]).to_dicts()[0]
        df_entry = df.slice(id, 1).select(["id", "title", "url"]).to_dicts()[0]
        log.info(f"search result id {id} corresponds to doc_id {df_entry['id']}, title {df_entry['title']}, url {df_entry['url']}")
        return str(df_entry['id'])

    # Translate raw_doc_ids to df_entry['id'] as strings
    translated_ids = [
        [get_doc_id(id, df) for id in id_rows]
        for id_rows in all_raw_doc_ids
    ]

    # Optional: Log the translated IDs
    log.info(f"Translated IDs: {translated_ids}")

    # given the raw doc_id, reverse the mapping to get the actual doc_id and title
    # calculate the offset of the parquet file, and read the correponding parquet file and the record
    # for id_rows in raw_doc_ids:
    #     for id in id_rows:
    #         assert id >= 0, "Doc ID should be non-negative"
    #         file_number = id // 100000 # TODO: do not hardcode 100K
    #         record_index = id % 100000
    #         log.info(f"Record index: {record_index}, File number: {file_number}")
    #         df_sub = pl.read_parquet(f'{args.embed_src_dir}/{file_number:03d}.parquet')
    #         # get the record_index th entry in the dataframe
    #         df_entry = df_sub.slice(record_index, 1).select(["id", "title", "url"]).to_dicts()[0]
    #         log.info(f"search result id {id} corresponds to doc_id {df_entry['id']}, title {df_entry['title']}, url {df_entry['url']}")
    # for qrel in dataset.qrels_iter():
    #     log.info('qrel: %s', qrel)
    #     break

