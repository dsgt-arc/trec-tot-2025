import pickle
from collections import defaultdict

index1_path = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/splade_indexes/history_geography_half1_splade.pkl"
index2_path = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/splade_indexes/history_geography_half2_splade.pkl"
merged_index_path = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/splade_indexes/history_geography_merged_splade.pkl"

def merge_splade_indexes(index1_path, index2_path, merged_index_path):
    # Load only metadata and corpus_ids first
    with open(index1_path, "rb") as f1, open(index2_path, "rb") as f2:
        index1 = pickle.load(f1)
        index2 = pickle.load(f2)
        corpus_ids1 = index1["corpus_ids"]
        corpus_ids2 = index2["corpus_ids"]
        threshold = index1["threshold"]

    merged_corpus_ids = corpus_ids1 + corpus_ids2
    offset = len(corpus_ids1)

    # Generator for all term_ids
    all_term_ids = set()
    with open(index1_path, "rb") as f1:
        index1 = pickle.load(f1)
        all_term_ids.update(index1["inverted_index"].keys())
    with open(index2_path, "rb") as f2:
        index2 = pickle.load(f2)
        all_term_ids.update(index2["inverted_index"].keys())

    # Merge inverted index term-by-term
    merged_inverted_index = {}
    for term_id in all_term_ids:
        postings = []
        # Get postings from index1
        with open(index1_path, "rb") as f1:
            index1 = pickle.load(f1)
            if term_id in index1["inverted_index"]:
                postings.extend(index1["inverted_index"][term_id])
        # Get postings from index2 (with offset)
        with open(index2_path, "rb") as f2:
            index2 = pickle.load(f2)
            if term_id in index2["inverted_index"]:
                postings.extend([(doc_idx + offset, weight) for doc_idx, weight in index2["inverted_index"][term_id]])
        # Sort postings by weight descending
        postings.sort(key=lambda x: x[1], reverse=True)
        merged_inverted_index[term_id] = postings

    merged_index = {
        "inverted_index": merged_inverted_index,
        "corpus_ids": merged_corpus_ids,
        "vocab_size": len(merged_inverted_index),
        "num_docs": len(merged_corpus_ids),
        "threshold": threshold,
    }

    # Save merged index
    with open(merged_index_path, "wb") as f:
        pickle.dump(merged_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Merged SPLADE index saved to {merged_index_path}")
    print(f"Vocab Size: {len(merged_inverted_index)}, Num Docs: {len(merged_corpus_ids)}")

if __name__ == "__main__":
    # merge_splade_indexes(index1_path, index2_path, merged_index_path)

    # open the merged index to verify
    print("Verifying merged index...")
    with open(merged_index_path, "rb") as f:
        merged_index = pickle.load(f)
        print(f"Loaded Merged Index - Vocab Size: {merged_index['vocab_size']}, Num Docs: {merged_index['num_docs']}")