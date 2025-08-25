import random
import os

def parse_trec_file(file_path):
    """Parse trec_val format file and return a dictionary of qid to doc_ids."""
    qid_to_docs = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid, doc_id = parts[0], parts[2]
            if qid not in qid_to_docs:
                qid_to_docs[qid] = []
            qid_to_docs[qid].append(doc_id)
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

def create_training_samples(dense_path, sparse_path, qrel_path, output_path, output_trec):
    dense_results = parse_trec_file(dense_path)
    sparse_results = parse_trec_file(sparse_path)
    qrel_data = parse_qrel_file(qrel_path)

    with open(output_path, 'w') as out_file:
        with open(output_trec, 'w') as trec_file:
            for qid in qrel_data:
                pos = 1
                relevant_docs = qrel_data[qid]
                dense_top10 = dense_results.get(qid, [])[:10]
                sparse_top10 = sparse_results.get(qid, [])[:10]

                # Ground truth documents with relevance score 2
                for doc_id in relevant_docs:
                    out_file.write(f"2 qid:{qid} doc:{doc_id}\n")
                    trec_file.write(f"{qid} Q0 {doc_id} {pos} 2.0 sampling\n")
                    pos += 1

                # Combine top-10 from dense and sparse, sample 5 pseudo-relevant docs with score 1
                combined_top20 = list(set(dense_top10 + sparse_top10))
                pseudo_relevant_docs = random.sample(combined_top20, min(5, len(combined_top20)))
                for doc_id in pseudo_relevant_docs:
                    if doc_id not in relevant_docs:
                        out_file.write(f"1 qid:{qid} doc:{doc_id}\n")
                        trec_file.write(f"{qid} Q0 {doc_id} {pos} 1.0 sampling\n")
                        pos += 1

                # Sample 10 irrelevant docs with score 0 from the rest of dense and sparse results
                remaining_docs = list(
                    set(dense_results.get(qid, [])[10:] + sparse_results.get(qid, [])[10:])  # Exclude top-10 from both dense and sparse
                    - set(relevant_docs)
                    - set(pseudo_relevant_docs)
                )
                irrelevant_docs = random.sample(remaining_docs, min(10, len(remaining_docs)))
                for doc_id in irrelevant_docs:
                    out_file.write(f"0 qid:{qid} doc:{doc_id}\n")
                    trec_file.write(f"{qid} Q0 {doc_id} {pos} 0.0 sampling\n")
                    pos += 1

if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH")
    dense_location = f"{DATA_PATH}/results/dev3-100/bge-filtered.txt"
    sparse_location = f"{DATA_PATH}/results/dev3-100/bm25.txt"
    qrel_location = f"{DATA_PATH}/2025/dev3-2025/qrel-first-100.txt"
    output_file = "outputs/dev3-100-sample.txt"
    output_trec = "outputs/dev3-100-sample-trec.txt"
    random.seed(42)
    create_training_samples(dense_location, sparse_location, qrel_location, output_file, output_trec)