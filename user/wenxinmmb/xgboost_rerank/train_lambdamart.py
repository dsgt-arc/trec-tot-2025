import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import ndcg_score
import argparse
import json  # Add this import

parser = argparse.ArgumentParser(description="Train a LambdaMART model.")
parser.add_argument("--dir", type=str, required=True, help="Working directory for outputs")
args = parser.parse_args()
# Define file paths
features_file = f"{args.dir}/features.txt"
model_file = f"{args.dir}/lambdamart_model.json"

# Load data from features file
def load_data(features_file):
    labels = []
    qids = []
    data = []
    with open(features_file, "r") as f:
        for line in f:
            # Ignore everything after the '#' symbol
            line = line.split("#")[0].strip()
            # Skip empty lines
            if not line:
                continue
            parts = line.strip().split()
            labels.append(float(parts[0]))  # Relevance label
            qid = parts[1].split(":")[1]  # Extract qid
            qids.append(int(qid))
            features = [float(part.split(":")[1]) for part in parts[2:]]
            data.append(features)
    return np.array(data), np.array(labels), np.array(qids)

# Group queries for LambdaMART
def group_queries(qids):
    groups = []
    current_qid = qids[0]
    count = 0
    for qid in qids:
        if qid == current_qid:
            count += 1
        else:
            groups.append(count)
            current_qid = qid
            count = 1
    groups.append(count)
    return groups

# Calculate NDCG and Recall
def calculate_metrics(labels, preds, qids, k_values):
    unique_qids = np.unique(qids)
    ndcg_scores = []
    recall_scores = {k: [] for k in k_values}

    for qid in unique_qids:
        indices = np.where(qids == qid)[0]
        true_relevance = labels[indices]
        predicted_scores = preds[indices]

        # Sort by predicted scores in descending order
        sorted_indices = np.argsort(-predicted_scores)
        true_relevance = true_relevance[sorted_indices]

        # Calculate NDCG
        ndcg_scores.append(ndcg_score([true_relevance], [predicted_scores[sorted_indices]], k=max(k_values)))

        # Calculate Recall@k
        for k in k_values:
            top_k_relevance = true_relevance[:k]
            recall = np.sum(top_k_relevance > 0) / np.sum(labels[indices] > 0)
            recall_scores[k].append(recall)

    avg_ndcg = np.mean(ndcg_scores)
    avg_recall = {k: np.mean(recall_scores[k]) for k in k_values}
    return avg_ndcg, avg_recall

# Load the data
data, labels, qids = load_data(features_file)
groups = group_queries(qids)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(data, label=labels)
dtrain.set_group(groups)

# Define LambdaMART parameters
params = {
    "objective": "rank:pairwise",  # Pairwise ranking objective
    "eval_metric": "ndcg",         # Evaluation metric
    "eta": 0.1,                    # Learning rate
    "max_depth": 6,                # Maximum tree depth
    "min_child_weight": 1,         # Minimum sum of instance weight (hessian) needed in a child
    "gamma": 0.0,                  # Minimum loss reduction required to make a further partition
    "subsample": 0.8,              # Subsample ratio of the training instances
    "colsample_bytree": 0.8        # Subsample ratio of columns when constructing each tree
}

# Evaluate before training
print("Evaluating before training...")
initial_preds = np.zeros_like(labels)  # Initial predictions are all zeros
k_values = [10, 100]
initial_ndcg, initial_recall = calculate_metrics(labels, initial_preds, qids, k_values)
print(f"Initial NDCG: {initial_ndcg}")
for k in k_values:
    print(f"Initial Recall@{k}: {initial_recall[k]}")

# Train the LambdaMART model
print("\nTraining LambdaMART model...")
num_rounds = 100
bst = xgb.train(params, dtrain, num_boost_round=num_rounds)

# Save the model
os.makedirs(os.path.dirname(model_file), exist_ok=True)
bst.save_model(model_file)
print(f"Model saved to {model_file}")

# Evaluate after training
print("\nEvaluating after training...")
final_preds = bst.predict(dtrain)
final_ndcg, final_recall = calculate_metrics(labels, final_preds, qids, k_values)
print(f"Final NDCG: {final_ndcg}")
for k in k_values:
    print(f"Final Recall@{k}: {final_recall[k]}")

# Save metrics to a JSON file
stats_file = f"{args.dir}/model_stats.json"
stats = {
    "initial": {
        "ndcg": initial_ndcg,
        "recall": {f"Recall@{k}": initial_recall[k] for k in k_values}
    },
    "final": {
        "ndcg": final_ndcg,
        "recall": {f"Recall@{k}": final_recall[k] for k in k_values}
    }
}

with open(stats_file, "w") as f:
    json.dump(stats, f, indent=4)
print(f"Model statistics saved to {stats_file}")