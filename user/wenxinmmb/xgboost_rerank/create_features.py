import os
import json
import argparse
from collections import defaultdict

# Global variables to cache data loaders
_pageview_cache = {}
_pagerank_cache = {}

# Define min and max values for normalization
QUERY_WORD_COUNT_MIN = 10
QUERY_WORD_COUNT_MAX = 1000
SPARSE_SCORE_MIN = 0
SPARSE_SCORE_MAX = 100

def load_pageview_data():
    """Load normalized pageview data from JSON file, with caching."""
    global _pageview_cache
    if _pageview_cache:
        return _pageview_cache

    pageview_file = os.path.join('outputs', 'normalized-features', 'normalized_pageview.json')
    with open(pageview_file, 'r') as f:
        _pageview_cache = json.load(f)
    print(f"Loaded {len(_pageview_cache)} normalized pageview records from {pageview_file}")

    return _pageview_cache

def load_pagerank_data():
    """Load normalized pagerank data from JSON file, with caching."""
    global _pagerank_cache
    if _pagerank_cache:
        return _pagerank_cache

    pagerank_file = os.path.join('outputs', 'normalized-features', 'normalized_pagerank.json')
    with open(pagerank_file, 'r') as f:
        _pagerank_cache = json.load(f)
    print(f"Loaded {len(_pagerank_cache)} normalized pagerank records from {pagerank_file}")

    return _pagerank_cache

# Parse arguments
parser = argparse.ArgumentParser(description="Feature extraction for RankLib training.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input file (sample or retrieval).")
parser.add_argument("--input_mode", type=str, choices=["sample", "retrieval", "sample-precomputed"], required=True, help="Mode of input file: 'sample', 'retrieval', or 'sample-precomputed'.")
parser.add_argument("--dense_feature_file", type=str, required=True, help="Path to the dense feature file.")
parser.add_argument("--sparse_feature_file", type=str, required=True, help="Path to the sparse feature file.")
parser.add_argument("--query_files", type=str, nargs='+', required=True, help="Paths to one or more query.jsonl files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
args = parser.parse_args()

# Define file paths
input_file = args.input_file
dense_feature_file = args.dense_feature_file
sparse_feature_file = args.sparse_feature_file
query_files = args.query_files
output_dir = args.output_dir
input_mode = args.input_mode

features_file = os.path.join(output_dir, "features.txt")
info_file = os.path.join(output_dir, "info.json")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load queries from multiple files
queries = {}
for qfile in query_files:
    with open(qfile, "r") as f:
        for line in f:
            query_data = json.loads(line.strip())
            word_count = len(query_data["query"].split())
            queries[query_data["query_id"]] = word_count

# Load dense scores (skip if sample-precomputed)
dense_scores = {}
sparse_scores = {}
if input_mode != "sample-precomputed":
    with open(dense_feature_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            qid, doc_id, score = parts[0], parts[2], float(parts[4])
            dense_scores[(qid, doc_id)] = score

    with open(sparse_feature_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            qid, doc_id, score = parts[0], parts[2], float(parts[4])
            sparse_scores[(qid, doc_id)] = score

# Load pageview and pagerank data
pageview_data = load_pageview_data()
pagerank_data = load_pagerank_data()

# Process input file and write features
feature_stats = defaultdict(list)
with open(input_file, "r") as f_in, open(features_file, "w") as f_out:
    for line in f_in:
        parts = line.strip().split()
        if input_mode == "sample":
            relevance, qid, doc_id = parts[0], parts[1][len('qid:'):], parts[2][len('doc:'):]
            dense_score = dense_scores[(qid, doc_id)]
            sparse_score = sparse_scores[(qid, doc_id)]
        elif input_mode == "retrieval":
            relevance = 0  # Dummy relevance for retrieval mode
            qid, doc_id = parts[0], parts[2]
            dense_score = dense_scores[(qid, doc_id)]
            sparse_score = sparse_scores[(qid, doc_id)]
        elif input_mode == "sample-precomputed":
            relevance, qid, doc_id = parts[0], parts[1][len('qid:'):], parts[2][len('doc:'):]
            dense_score = float(parts[3][len('dense_score:'):])
            sparse_score = float(parts[4][len('sparse_score:'):])

        # Normalize query word count
        query_word_count = queries[qid]
        normalized_query_word_count = (query_word_count - QUERY_WORD_COUNT_MIN) / (QUERY_WORD_COUNT_MAX - QUERY_WORD_COUNT_MIN)
        normalized_query_word_count = max(0, min(1, normalized_query_word_count))  # Clamp to [0, 1]

        # Normalize sparse score
        normalized_sparse_score = (sparse_score - SPARSE_SCORE_MIN) / (SPARSE_SCORE_MAX - SPARSE_SCORE_MIN)
        normalized_sparse_score = max(0, min(1, normalized_sparse_score))  # Clamp to [0, 1]

        # Get pageview and pagerank
        pageview = pageview_data.get(doc_id, 0)
        pagerank = pagerank_data.get(doc_id, 0.0)

        # Append features to stats
        feature_stats["dense"].append(dense_score)
        feature_stats["sparse"].append(normalized_sparse_score)
        feature_stats["query_word_count"].append(normalized_query_word_count)
        feature_stats["pageview"].append(pageview)
        feature_stats["pagerank"].append(pagerank)

        # Write features to file
        f_out.write(f"{relevance} qid:{qid} 1:{dense_score} 2:{normalized_sparse_score} 3:{normalized_query_word_count} 4:{pageview} 5:{pagerank} # {doc_id}\n")

# Write info.json
info = {
    "description": "Features extracted for RankLib training",
    "features": {
        "1": f"Dense score from {os.path.basename(dense_feature_file)} (raw)",
        "2": f"Sparse score from {os.path.basename(sparse_feature_file)} (min-max normalized in range [0-100])",
        "3": "Query word count (min-max normalized)",
        "4": "Pageview count (log1p + z score normalized)",
        "5": "Pagerank score (log + z score normalized)"
    }
}
with open(info_file, "w") as f:
    json.dump(info, f, indent=4)

# Print feature statistics
dense_range = (min(feature_stats["dense"]), max(feature_stats["dense"]))
sparse_range = (min(feature_stats["sparse"]), max(feature_stats["sparse"]))
query_word_count_range = (min(feature_stats["query_word_count"]), max(feature_stats["query_word_count"]))
pageview_range = (min(feature_stats["pageview"]), max(feature_stats["pageview"]))
pagerank_range = (min(feature_stats["pagerank"]), max(feature_stats["pagerank"]))
print(f"Dense score range: {dense_range}")
print(f"Sparse score range: {sparse_range}")
print(f"Query word count range: {query_word_count_range}")
print(f"Pageview range: {pageview_range}")
print(f"Pagerank range: {pagerank_range}")
