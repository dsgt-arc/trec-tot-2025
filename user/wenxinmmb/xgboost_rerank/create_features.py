import os
import json
from collections import defaultdict

# Define file paths
sample_file = "outputs/dev3-100-sample.txt"
dense_file = "outputs/run6-sample-dev3-bge-dense.txt"
sparse_file = "outputs/run6-sample-dev3-pt.txt"
output_dir = "outputs/sample-v1/"
features_file = os.path.join(output_dir, "features.txt")
info_file = os.path.join(output_dir, "info.json")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load dense scores
dense_scores = {}
with open(dense_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        qid, doc_id, score = parts[0], parts[2], float(parts[4])
        dense_scores[(qid, doc_id)] = score

# Load sparse scores
sparse_scores = {}
with open(sparse_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        qid, doc_id, score = parts[0], parts[2], float(parts[4])
        sparse_scores[(qid, doc_id)] = score

# Process samples and write features
feature_stats = defaultdict(list)
with open(sample_file, "r") as f_in, open(features_file, "w") as f_out:
    for line in f_in:
        parts = line.strip().split()
        relevance, qid, doc_id = parts[0], parts[1][len('qid:'):], parts[2][len('doc:'):]
        dense_score = dense_scores[(qid,doc_id)]
        sparse_score = sparse_scores[(qid,doc_id)]
        feature_stats["dense"].append(dense_score)
        feature_stats["sparse"].append(sparse_score)
        f_out.write(f"{relevance} qid:{qid} 1:{dense_score} 2:{sparse_score} # {doc_id}\n")

# Write info.json
info = {
    "description": "Features extracted for RankLib training",
    "features": {
        "1": "Dense score from run6-sample-dev3-bge-dense.txt",
        "2": "Sparse score from run6-sample-dev3-pt.txt"
    }
}
with open(info_file, "w") as f:
    json.dump(info, f, indent=4)

# Print feature statistics
dense_range = (min(feature_stats["dense"]), max(feature_stats["dense"]))
sparse_range = (min(feature_stats["sparse"]), max(feature_stats["sparse"]))
print(f"Dense score range: {dense_range}")
print(f"Sparse score range: {sparse_range}")

# Suggest normalization
print("\nSuggested normalization:")
print("1. Min-max normalization: (x - min) / (max - min)")
print("2. Z-score normalization: (x - mean) / std")