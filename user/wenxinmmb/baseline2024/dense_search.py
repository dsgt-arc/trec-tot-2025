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

log = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("dense_search", description="semantic search on pre-computed embeddings found on huggingface")
    parser.add_argument("--hf_dataset", default="Upstash/wikipedia-2024-06-bge-m3", help="the huggingface embedding dataset to use")
    parser.add_argument("--data_version", default="2025",
                        help="data version to use, e.g., 2024 or 2025")
    parser.add_argument("--data_path", default="./datasets/TREC-ToT2024/", help="location to dataset")
    
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

    # download wikipedia embeddings
    dataset = load_dataset("Upstash/wikipedia-2024-06-bge-m3", "en", split="train",
                           streaming=True)
    row = 0
    all_embeddings = []
    for data in dataset:
        data_id = data["id"]
        url = data["url"]
        title = data["title"]
        text = data["text"]
        embedding = data["embedding"]
        all_embeddings.append(embedding)
        row += 1
        if row > 10:
            break
    
    # insert the data in FAISS
    embedding_size = 1024
    num_embeddings = len(all_embeddings)
    index = faiss.IndexFlatIP(embedding_size) # this uses inner product (dot product) for similarity
    indexwmap = faiss.IndexIDMap(index)
    all_embeddings = np.asarray(all_embeddings) # embedding is float64
    indexwmap.add_with_ids(all_embeddings, np.arange(num_embeddings, dtype=np.int64))

    # load encoder and calculate query embeddings
    transformer = SentenceTransformer(
        "BAAI/bge-m3",
        device=device,
        revision="babcf60cae0a1f438d7ade582983d4ba462303c2",
    )

    dataset = ir_datasets.load("trec-tot:train-2024")
    for q in dataset.queries_iter():
        log.info('query: %s', q)
        query_vector = transformer.encode(
            sentences=[q.query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        scores, raw_doc_ids = index.search(query_vector, k=1)
        log.info(f"Query: {q.query}, Score: {scores[0][0]}, Doc ID: {raw_doc_ids[0][0]}")
        break

    for qrel in dataset.qrels_iter():
        log.info('qrel: %s', qrel)
        break

