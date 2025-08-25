import os
import json
import argparse
from collections import defaultdict

# Parse arguments
parser = argparse.ArgumentParser(description="Feature extraction for RankLib training.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input file (sample or retrieval).")
parser.add_argument("--input_mode", type=str, choices=["sample", "retrieval"], required=True, help="Mode of input file: 'sample' or 'retrieval'.")
parser.add_argument("--dense_feature_file", type=str, required=True, help="Path to the dense feature file.")
parser.add_argument("--sparse_feature_file", type=str, required=True, help="Path to the sparse feature file.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
args = parser.parse_args()

# Define file paths
input_file = args.input_file
dense_feature_file = args.dense_feature_file
sparse_feature_file = args.sparse_feature_file
output_dir = args.output_dir
input_mode = args.input_mode

features_file = os.path.join(output_dir, "features.txt")
info_file = os.path.join(output_dir, "info.json")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load dense scores
dense_scores = {}
with open(dense_feature_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        qid, doc_id, score = parts[0], parts[2], float(parts[4])
        dense_scores[(qid, doc_id)] = score

# Load sparse scores
sparse_scores = {}
with open(sparse_feature_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        qid, doc_id, score = parts[0], parts[2], float(parts[4])
        sparse_scores[(qid, doc_id)] = score

# Process input file and write features
feature_stats = defaultdict(list)
with open(input_file, "r") as f_in, open(features_file, "w") as f_out:
    for line in f_in:
        parts = line.strip().split()
        if input_mode == "sample":
            relevance, qid, doc_id = parts[0], parts[1][len('qid:'):], parts[2][len('doc:'):]
        elif input_mode == "retrieval":
            relevance = 0  # Dummy relevance for retrieval mode
            qid, doc_id = parts[0], parts[2]
        dense_score = dense_scores[(qid, doc_id)]
        sparse_score = sparse_scores[(qid, doc_id)]
        feature_stats["dense"].append(dense_score)
        feature_stats["sparse"].append(sparse_score)
        f_out.write(f"{relevance} qid:{qid} 1:{dense_score} 2:{sparse_score} # {doc_id}\n")

# Write info.json
info = {
    "description": "Features extracted for RankLib training",
    "features": {
        "1": f"Dense score from {os.path.basename(dense_feature_file)}",
        "2": f"Sparse score from {os.path.basename(sparse_feature_file)}"
    }
}
with open(info_file, "w") as f:
    json.dump(info, f, indent=4)

# Print feature statistics
dense_range = (min(feature_stats["dense"]), max(feature_stats["dense"]))
sparse_range = (min(feature_stats["sparse"]), max(feature_stats["sparse"]))
print(f"Dense score range: {dense_range}")
print(f"Sparse score range: {sparse_range}")
