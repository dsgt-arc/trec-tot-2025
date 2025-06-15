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


def do_range_search(fl_pattern, res_name):
        
        # Initialize FAISS index
        embedding_size = 1024
        index = faiss.IndexFlatIP(embedding_size) # this uses inner product (dot product) for similarity

        # Read all files into a single DataFrame
        directory = args.embed_src_dir
        file_pattern = f"{directory}/{fl_pattern}"
        file_paths = glob.glob(file_pattern)
        df = pl.read_parquet(file_paths)

        embeddings = np.vstack(df['embedding'].to_numpy())
        # indexwmap.add_with_ids(embeddings, ids)
        index.add(embeddings)
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
        qids = []
        for q in dataset.queries_iter():
            # log first a few characters of the query
            log.info('Processing query: %s %s', q.query_id, q.query[:150])
            queries.append(q.query)
            qids.append(q.query_id)
            if len(queries) >= encode_batch_size:
                query_vectors = transformer.encode(
                    sentences=queries,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                )
                scores, raw_doc_ids = index.search(query_vectors, k=top_k)
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
            scores, raw_doc_ids = index.search(query_vectors, k=top_k)
            log.info(f"Score: \n{scores}, Doc ID: \n{raw_doc_ids}")
            # Aggregate scores and raw_doc_ids
            all_scores = np.vstack((all_scores, scores))
            all_raw_doc_ids = np.vstack((all_raw_doc_ids, raw_doc_ids))

        # print the shape of the all_scores and raw_doc_ids
        log.info(f"Total queries processed: scores shape -- {all_scores.shape}, raw_doc_ids shape -- {all_raw_doc_ids.shape}")
        
        def get_doc_id(id, df):
            assert id >= 0, "Doc ID should be non-negative"
            df_entry = df.slice(id, 1).select(["id", "title", "url"]).to_dicts()[0]
            log.info(f"search result id {id} --> doc_id {df_entry['id']}, title {df_entry['title']}")
            return str(df_entry['id'])

        # Translate raw_doc_ids to df_entry['id'] as strings
        translated_ids = [
            [get_doc_id(id, df) for id in id_rows]
            for id_rows in all_raw_doc_ids
        ]

        # write the results to a file
        with open(res_name, 'a') as f:
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

    # do_range_search('0[6-9][0-9].parquet', 'search_results_060_099.json')
    # do_range_search('1[0-5][0-9].parquet', 'search_results_100_159.json')
    # do_range_search('1[6-9][0-9].parquet', 'search_results_160_199.json')
    do_range_search('2[0-5][0-9].parquet', 'search_results_200_259.json')
    do_range_search('2[6-9][0-9].parquet', 'search_results_260_299.json')
    do_range_search('3[0-5][0-9].parquet', 'search_results_300_359.json')
    do_range_search('3[6-9][0-9].parquet', 'search_results_360_399.json')
    do_range_search('4[0-5][0-9].parquet', 'search_results_400_459.json')
    do_range_search('4[6-9][0-9].parquet', 'search_results_460_499.json')
