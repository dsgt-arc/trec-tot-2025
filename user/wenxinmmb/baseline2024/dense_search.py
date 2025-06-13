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
from itertools import islice

log = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("dense_search", description="semantic search on pre-computed embeddings found on huggingface")
    parser.add_argument("--hf_dataset", default="Upstash/wikipedia-2024-06-bge-m3", help="the huggingface embedding dataset to use")
    parser.add_argument("--data_version", default="2025",
                        help="data version to use, e.g., 2024 or 2025")
    parser.add_argument("--data_path", default="./datasets/TREC-ToT2024/", help="location to dataset")
    parser.add_argument("--polar_dir", default="./polar_dir/embed_1", help="location to store embeddings")
    
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

    # The cohere model isn't open-sourced :(
    # transformer = SentenceTransformer(
    #     "Cohere/Cohere-embed-multilingual-v3.0", 
    #     device=device
    # )

    args = parser.parse_args()
    if args.data_version == "2025":
        import tot_25 as tot
    elif args.data_version == "2024":
        import tot_24 as tot
    else:
        raise ValueError(f"Unknown data version: {args.data_version}")

    tot.register(args.data_path)

    os.makedirs(args.polar_dir, exist_ok=True)

    chunk_size = 100000 # number of embeddings to process at a time
    total_records = chunk_size*10  # max = 47_018_430
    chunk_data = []
    chunk_embeddings = []
    chunk_index = 0

    # Initialize FAISS index
    # embedding_size = 1024
    # index = faiss.IndexFlatIP(embedding_size) # this uses inner product (dot product) for similarity
    # indexwmap = faiss.IndexIDMap(index)

    # download wikipedia embeddings
    # dataset = load_dataset("Upstash/wikipedia-2024-06-bge-m3", "en", split="train")

    # There are 470 parquet files in total, looks like 100K records per file
    parquet_directory= '/Users/wenxin/tot/tot_data/upstash-embed/*.parquet'
    df = pl.read_parquet(parquet_directory)
    # print df stats
    log.info(f"Loaded dataset with {df.shape[0]} records.")
    # print available fields in the dataframe
    log.info(f"Available fields: {df.columns}")
    # print first 5 records
    log.info(f"First 5 records:\n{df.head(5)}")
    
    for row, data in enumerate(dataset):
        # Extract metadata and embedding
        metadata_entry = {
            "int_id": row,
            "title": data["title"],
            "url": data["url"],
            "text": data["text"],
            "orig_id": data["id"],
        }
        embedding = np.asarray(data["embedding"], dtype=np.float32)

        # Append to chunk lists
        chunk_data.append(metadata_entry)
        chunk_embeddings.append(embedding)

        # Write chunk to Parquet when chunk_size is reached
        if len(chunk_data) >= chunk_size:
            # Write metadata chunk to Parquet
            metadata_df = pl.DataFrame(chunk_data)
            metadata_file = os.path.join(args.polar_dir, f"metadata_chunk_{chunk_index}.parquet")
            metadata_df.write_parquet(metadata_file)

            # Write embeddings chunk to a separate file (optional)
            embeddings_file = os.path.join(args.polar_dir, f"embeddings_chunk_{chunk_index}.npy")
            np.save(embeddings_file, np.array(chunk_embeddings))

            print(f"Written chunk {chunk_index} to disk.")
            
            # Reset chunk lists and increment chunk index
            chunk_data = []
            chunk_embeddings = []
            chunk_index += 1
        
    # Disable the following
    # all_embeddings = np.asarray(all_embeddings) # embedding is float64
    # indexwmap.add_with_ids(all_embeddings, np.arange(num_embeddings, dtype=np.int64))

    # # load encoder and calculate query embeddings
    # transformer = SentenceTransformer(
    #     "BAAI/bge-m3",
    #     device=device,
    #     revision="babcf60cae0a1f438d7ade582983d4ba462303c2",
    # )

    # dataset = ir_datasets.load("trec-tot:train-2024")
    # for q in dataset.queries_iter():
    #     log.info('query: %s', q)
    #     query_vector = transformer.encode(
    #         sentences=[q.query],
    #         show_progress_bar=False,
    #         normalize_embeddings=True,
    #     )
    #     scores, raw_doc_ids = index.search(query_vector, k=1)
    #     log.info(f"Query: {q.query}, Score: {scores[0][0]}, Doc ID: {raw_doc_ids[0][0]}")
    #     break

    # for qrel in dataset.qrels_iter():
    #     log.info('qrel: %s', qrel)
    #     break

