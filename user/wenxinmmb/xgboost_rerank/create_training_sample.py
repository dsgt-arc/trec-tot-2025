import random
import os
from tqdm import tqdm

def parse_trec_file(file_path):
    """Parse trec_val format file and return a dictionary of qid to list of (doc_id, score)."""
    qid_to_docs = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid, doc_id, score = parts[0], parts[2], float(parts[4])
            if qid not in qid_to_docs:
                qid_to_docs[qid] = []
            qid_to_docs[qid].append((doc_id, score))
    return qid_to_docs

def parse_qrel_file(qrel_path):
    """Parse qrel file and return a dictionary of qid to relevant doc_ids."""
    qid_to_relevant_docs = {}
    with open(qrel_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid, doc_id, relevance = parts[0], parts[2], int(parts[3])
            if relevance > 0:  # Only consider relevant documents
                if qid not in qid_to_relevant_docs:
                    qid_to_relevant_docs[qid] = set()
                qid_to_relevant_docs[qid].add(doc_id)
    return qid_to_relevant_docs


# Global cache for dense scores
dense_score_cache = {}

def load_dense_scores(tsv_path):
    """Load dense scores from TSV file into a global cache."""
    global dense_score_cache
    dense_score_cache = {}
    with open(tsv_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            doc_id, qid, score = parts[0], parts[1], float(parts[2])
            dense_score_cache[(doc_id, qid)] = score

def get_dense_score(doc_id, qid):
    """Fetch dense score from the global cache."""
    return dense_score_cache.get((doc_id, qid), 0.0)

def get_sparse_score_streaming(doc_id, qid, sparse_score_path):
    """
    Fetch sparse score for a given doc_id and qid from the sparse_score_path file.
    File format: qid Q0 doc_id rank score pt
    """
    with open(sparse_score_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[0] == str(qid) and parts[2] == str(doc_id):
                try:
                    return float(parts[4])
                except ValueError:
                    return 0.0
    return 0.0

def load_sparse_score(sparse_score_path):
    sparse_score_cache = {}
    with open(sparse_score_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            qid, doc_id, score = parts[0], parts[2], float(parts[4])
            sparse_score_cache[(doc_id, qid)] = score
    return sparse_score_cache

def get_sparse_score(doc_id, qid, sparse_score_cache):
    return sparse_score_cache.get((doc_id, qid), 0.0)

def create_training_samples(qrel_path, dense_path, sparse_path, sparse_score_path, qrel_sparse_score_path, output_path, output_trec):
    dense_results = parse_trec_file(dense_path)
    sparse_results = parse_trec_file(sparse_path)
    qrel_data = parse_qrel_file(qrel_path)
    sparse_score_cache = load_sparse_score(sparse_score_path)
    qrel_sparse_score_cache = load_sparse_score(qrel_sparse_score_path)

    with open(output_path, 'a') as out_file:
        with open(output_trec, 'a') as trec_file:
            for qid in tqdm(qrel_data, desc="Processing queries"):
                pos = 1
                relevant_docs = qrel_data[qid]
                dense_top10 = dense_results.get(qid, [])[:10]
                sparse_top10 = sparse_results.get(qid, [])[:10]

                # Ground truth documents with relevance score 2
                for doc_id in relevant_docs:
                    # dense_score, sparse_score = get_dense_and_sparse_score(doc_id, qid, dense_score_path, sparse_score_path)
                    dense_score = get_dense_score(doc_id, qid)
                    # sparse_score = get_sparse_score_streaming(doc_id, qid, qrel_sparse_score_path)
                    sparse_score = get_sparse_score(doc_id, qid, qrel_sparse_score_cache)

                    out_file.write(f"2 qid:{qid} doc:{doc_id} dense_score:{dense_score} sparse_score:{sparse_score}\n")
                    trec_file.write(f"{qid} Q0 {doc_id} {pos} 2.0 sampling\n")
                    pos += 1

                # Combine top-10 from dense and sparse, sample 5 pseudo-relevant docs with score 1
                combined_top20 = list(set(dense_top10 + sparse_top10))
                pseudo_relevant_docs = random.sample(combined_top20, min(5, len(combined_top20)))
                
                combined_top20 = {}
                for doc_id, score in dense_top10:
                    combined_top20[doc_id] = (doc_id, score, 'dense')
                for doc_id, score in sparse_top10:
                    combined_top20[doc_id] = (doc_id, score, 'sparse')
                # remove doc that is in relevant_docs
                for doc_id in relevant_docs:
                    combined_top20.pop(doc_id, None)
                # randomly select 5 in combined_top20
                pseudo_relevant_docs = random.sample(list(combined_top20.keys()), min(5, len(combined_top20)))

                for doc_id in pseudo_relevant_docs:
                    assert (doc_id not in relevant_docs)
                    label = combined_top20[doc_id][2]
                    if label == 'dense':
                        dense_score = combined_top20[doc_id][1]
                    else:
                        dense_score = get_dense_score(doc_id, qid)
                    
                    if label == 'sparse':
                        sparse_score = combined_top20[doc_id][1]
                    else:
                        # sparse_score = get_sparse_score_streaming(doc_id, qid, sparse_score_path)
                        sparse_score = get_sparse_score(doc_id, qid, sparse_score_cache)

                    out_file.write(f"1 qid:{qid} doc:{doc_id} dense_score:{dense_score} sparse_score:{sparse_score}\n")
                    trec_file.write(f"{qid} Q0 {doc_id} {pos} 1.0 sampling\n")
                    pos += 1

                # Sample 10 irrelevant docs with score 0 from the rest of dense and sparse results
                dense_remain = dense_results.get(qid, [])[10:]
                sparse_remain = sparse_results.get(qid, [])[10:]

                # Build a dict to track source and score
                remaining_docs_dict = {}
                for doc_id, score in dense_remain:
                    if doc_id not in relevant_docs and doc_id not in pseudo_relevant_docs:
                        remaining_docs_dict[doc_id] = ('dense', score)
                for doc_id, score in sparse_remain:
                    if doc_id not in relevant_docs and doc_id not in pseudo_relevant_docs:
                        # If already present from dense, keep dense; else add sparse
                        if doc_id not in remaining_docs_dict:
                            remaining_docs_dict[doc_id] = ('sparse', score)

                irrelevant_doc_ids = random.sample(list(remaining_docs_dict.keys()), min(10, len(remaining_docs_dict)))
                for doc_id in irrelevant_doc_ids:
                    source, score = remaining_docs_dict[doc_id]
                    if source == 'dense':
                        dense_score = score
                        sparse_score = get_sparse_score(doc_id, qid, sparse_score_cache)
                    else:  # source == 'sparse'
                        dense_score = get_dense_score(doc_id, qid)
                        sparse_score = score

                    out_file.write(f"0 qid:{qid} doc:{doc_id} dense_score:{dense_score} sparse_score:{sparse_score}\n")
                    trec_file.write(f"{qid} Q0 {doc_id} {pos} 0.0 sampling\n")
                    pos += 1

if __name__ == "__main__":
    random.seed(42)
    DATA_PATH = os.getenv("DATA_PATH")
    TOT = os.getenv("TOT")

    # ====================
    # sample-v1
    # ====================
    # dense_location = f"{DATA_PATH}/results/dev3-100/bge-filtered.txt"
    # sparse_location = f"{DATA_PATH}/results/dev3-100/bm25.txt"
    # qrel_location = f"{DATA_PATH}/2025/dev3-2025/qrel-first-100.txt"
    # output_file = "outputs/dev3-100-sample.txt"
    # output_trec = "outputs/dev3-100-sample-trec.txt"
    # create_training_samples(dense_location, sparse_location, qrel_location, output_file, output_trec)

    # ====================
    # sample-v2
    # ====================
    # create training set from three sources
    # 1. train set  -- 143 queries
    # 2. dev3 first 200 -- 200 queries
    # 3. llmset1 train -- 4002 queries

    qrel_locations = [
        f"{DATA_PATH}/2025/train-2025/qrel.txt",
        f"{DATA_PATH}/2025/dev3-2025/qrel-first-200.txt",
        f"{DATA_PATH}/2025/llmset1-train-2025/qrel.txt"
    ]

    dense_retrieval_paths = [
        f"{DATA_PATH}/results/train/bge-filtered.txt",
        f"{DATA_PATH}/results/dev3/bge-filtered.txt",
        f"{DATA_PATH}/results/llmset1-train/bge-filtered.txt"
    ]

    sparse_retrieval_paths = [
        f"{DATA_PATH}/results/train/bm25.txt",
        f"{DATA_PATH}/results/dev3/bm25.txt",
        f"{DATA_PATH}/results/llmset1-train/bm25.txt"
    ]

    dense_score_location = f"{TOT}/baseline2024/dense/outputs/dense_score/all-sets.tsv"
    qrel_sparse_score_paths = [
        f"outputs/scores/qrel-sparse-train/qrel--md-pt.txt",
        f"outputs/scores/qrel-sparse-dev3/qrel--md-pt.txt",
        f"outputs/scores/qrel-sparse-llmset1-train/qrel--md-pt.txt"
    ]

    sparse_score_paths = [
        f"outputs/scores/train-bge/bge-filtered--md-pt.txt", # TODO: Pending
        f"outputs/scores/dev3-bge/bge-filtered--md-pt.txt",
        f"outputs/scores/llmset1-train-bge/bge-filtered--md-pt.txt" # TODO: Pending
    ]

    load_dense_scores(dense_score_location)
    output_dir = "outputs/sample-v4"
    # create the directory if not exist
    os.makedirs(output_dir, exist_ok=True)

    # change to range(3) once sparse score path is ready
    for i in range(3):
        create_training_samples(
            qrel_locations[i],
            dense_retrieval_paths[i],
            sparse_retrieval_paths[i],
            sparse_score_paths[i],
            qrel_sparse_score_paths[i],
            f"{output_dir}/sample.txt",
            f"{output_dir}/sample-trec.txt"
        )
