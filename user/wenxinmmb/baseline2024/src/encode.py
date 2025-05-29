import logging

import faiss
import ir_datasets
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from tqdm import tqdm, trange

from src import data
import pyarrow as pa
import pyarrow.parquet as pq
import os

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s - %(message)s')
log.setLevel(logging.INFO)

def embed_dataset(model: SentenceTransformer, dataset: ir_datasets.Dataset, device,
                          encode_batch_size, directory, max_embeddings_per_file,
                          normalize_embeddings=False ):
    """
    Embed the documents in the dataset using the provided model.
    and save the embeddings into parquet files under specified directory.
    """
    log.info("loading docs")
    doc_ids, documents = data.get_documents(dataset)
    model = model.eval().to(device)

    # check if directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        # if directory exists, skip embedding
        log.warning(f"Directory {directory} already exists! Skipping embedding.")
        return doc_ids


    def _save_embeddings_to_disk(embeddings, start_index):
        """
        Save embeddings to disk in parquet format.
        """
        # TODO: make parquet file self-contained, i.e., include doc_ids
        table = pa.Table.from_arrays([pa.array(embeddings)], names=["embeddings"])
        pq.write_table(table, os.path.join(directory, f"embeddings_{start_index:010}.parquet"))

    all_embeddings = []
    doc_length = len(documents)
    # doc_length = 10000 # FIJI: for testing a small set of record
    for start_index in trange(0, doc_length, encode_batch_size, desc="Batches"):
        end_index = min(start_index + encode_batch_size, doc_length)
        sentences_batch = documents[start_index:end_index]
        features = batch_to_device(model.tokenize(sentences_batch), device)

        with torch.no_grad():
            out_features = model.forward(features)
            embeddings = out_features["sentence_embedding"]
            embeddings = embeddings.detach()
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            embeddings = embeddings.cpu().numpy()

            all_embeddings.extend(embeddings)

        if (end_index) % max_embeddings_per_file == 0:
            assert(end_index != 0)
            log.info(f"Processed {end_index} documents, saving embeddings from {start_index} to {end_index} to disk")    
            _save_embeddings_to_disk(all_embeddings, start_index+encode_batch_size)
            all_embeddings = []

    # Save any remaining embeddings
    if len(all_embeddings) > 0:
        log.info(f"Processed {doc_length} documents, saving remaining embeddings to disk")
        _save_embeddings_to_disk(all_embeddings, doc_length)
    
    return doc_ids

def create_faiss_index(embedding_size: int, directory: str, doc_ids: list):
    """
    Create a FAISS index from the embeddings stored in parquet files.
    """
    log.info("Creating FAISS index from embeddings files")
    all_embeddings = []

    parquet_files = sorted([f for f in os.listdir(directory) if f.endswith('.parquet')])
    for parquet_file in tqdm(parquet_files, desc="Loading embeddings"):
        file_path = os.path.join(directory, parquet_file)
        table = pq.read_table(file_path)
        embeddings = table.column("embeddings").to_numpy()
        all_embeddings.extend(embeddings)
    num_embeddings = len(all_embeddings)
    all_embeddings = np.asarray(all_embeddings)
    index = faiss.IndexFlatIP(embedding_size) # this uses inner product (dot product) for similarity
    indexwmap = faiss.IndexIDMap(index)
    indexwmap.add_with_ids(all_embeddings, np.arange(num_embeddings, dtype=np.int64))

    if num_embeddings != len(doc_ids):
        log.warning(f"Number of embeddings ({num_embeddings}) does not match number of documents ({len(doc_ids)})!")

    idx_to_docid = {}
    docid_to_idx = {}
    for idx, doc_id in enumerate(doc_ids[0:num_embeddings]):
        idx_to_docid[idx] = doc_id
        docid_to_idx[doc_id] = idx
    return indexwmap, (idx_to_docid, docid_to_idx)


def create_run_faiss(model: SentenceTransformer, dataset: ir_datasets.Dataset, device, eval_batch_size,
                     index: faiss.IndexIDMap, idx_to_docid, top_k):
    model.eval()

    qids = []
    queries = []
    for query in dataset.queries_iter():
        queries.append(query.query)
        qids.append(query.query_id)

    with torch.no_grad():
        query_embeddings = model.encode(queries, batch_size=eval_batch_size, show_progress_bar=True,
                                        convert_to_numpy=True, device=device)

    os.environ["OMP_NUM_THREADS"] = "1" # FIXME: set OMP thread to 1, otherwise FAISS CPU search hit memory leak on mac os
    scores, raw_doc_ids = index.search(query_embeddings, k=top_k)
    run = {}

    log.info(f"processing FAISS search results")
    for qid, sc, rdoc_ids in zip(qids, scores, raw_doc_ids):
        run[qid] = {}
        for s, rdid in zip(sc, rdoc_ids):
            if rdid == -1:
                log.warning(f"invalid doc ids!")
                continue
            run[qid][idx_to_docid[rdid]] = float(s)

    return run


def create_qrel(dataset, run=None):
    qrel = {}
    n_missing = 0
    for q in dataset.qrels_iter():
        if run and q.query_id not in run:
            n_missing += 1
        qrel[q.query_id] = {q.doc_id: q.relevance}

    return qrel, n_missing
