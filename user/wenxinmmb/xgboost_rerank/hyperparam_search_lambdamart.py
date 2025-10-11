import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import ndcg_score
from sklearn.model_selection import ParameterGrid, train_test_split
import argparse
import json
import random

parser = argparse.ArgumentParser(description="Train a LambdaMART model.")
parser.add_argument("--dir", type=str, required=True, help="Working directory for outputs")
parser.add_argument("--search-mode",
    choices=["none", "grid", "random"],
    default="none",
    help="Hyperparameter search mode: none, grid, or random"
)
args = parser.parse_args()
# Define file paths
features_file = f"{args.dir}/features.txt"
model_file = f"{args.dir}/lambdamart_model.json"

def load_data(features_file):
    labels = []
    qids = []
    data = []
    with open(features_file, "r") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.strip().split()
            labels.append(float(parts[0]))
            qid = parts[1].split(":")[1]
            qids.append(int(qid))
            features = [float(part.split(":")[1]) for part in parts[2:]]
            data.append(features)
    return np.array(data), np.array(labels), np.array(qids)

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

def calculate_metrics(labels, preds, qids, k_values):
    unique_qids = np.unique(qids)
    ndcg_scores = []
    recall_scores = {k: [] for k in k_values}
    for qid in unique_qids:
        indices = np.where(qids == qid)[0]
        true_relevance = labels[indices]
        predicted_scores = preds[indices]
        sorted_indices = np.argsort(-predicted_scores)
        true_relevance = true_relevance[sorted_indices]
        ndcg_scores.append(ndcg_score([true_relevance], [predicted_scores[sorted_indices]], k=max(k_values)))
        for k in k_values:
            top_k_relevance = true_relevance[:k]
            recall = np.sum(top_k_relevance > 0) / np.sum(labels[indices] > 0)
            recall_scores[k].append(recall)
    avg_ndcg = np.mean(ndcg_scores)
    avg_recall = {k: np.mean(recall_scores[k]) for k in k_values}
    return avg_ndcg, avg_recall

data, labels, qids = load_data(features_file)
k_values = [10, 100]

if args.search_mode == "grid":
    print("Grid search enabled. Splitting train/validation...")
    # Split by queries (qid) to avoid leakage
    unique_qids = np.unique(qids)
    train_qids, val_qids = train_test_split(unique_qids, test_size=0.2, random_state=42)
    train_idx = np.isin(qids, train_qids)
    val_idx = np.isin(qids, val_qids)

    train_data, train_labels, train_qids_arr = data[train_idx], labels[train_idx], qids[train_idx]
    val_data, val_labels, val_qids_arr = data[val_idx], labels[val_idx], qids[val_idx]

    train_groups = group_queries(train_qids_arr)
    val_groups = group_queries(val_qids_arr)

    dtrain = xgb.DMatrix(train_data, label=train_labels)
    dtrain.set_group(train_groups)
    dval = xgb.DMatrix(val_data, label=val_labels)
    dval.set_group(val_groups)

    param_grid = {
        "eta": [0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    num_rounds = 100
    best_ndcg = -1
    best_params = None
    best_model = None

    for grid_params in ParameterGrid(param_grid):
        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            **grid_params
        }
        bst = xgb.train(params, dtrain, num_boost_round=num_rounds)
        val_preds = bst.predict(dval)
        val_ndcg, _ = calculate_metrics(val_labels, val_preds, val_qids_arr, k_values)
        print(f"Params: {grid_params}, Validation NDCG: {val_ndcg}")
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_params = grid_params
            best_model = bst

    print(f"Best Params: {best_params}, Best Validation NDCG: {best_ndcg}")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    best_model.save_model(model_file)
    print(f"Best model saved to {model_file}")

    # Evaluate on train and val
    train_preds = best_model.predict(dtrain)
    train_ndcg, train_recall = calculate_metrics(train_labels, train_preds, train_qids_arr, k_values)
    val_preds = best_model.predict(dval)
    val_ndcg, val_recall = calculate_metrics(val_labels, val_preds, val_qids_arr, k_values)

    stats_file = f"{args.dir}/model_stats_gridsearch.json"
    stats = {
        "best_params": best_params,
        "train": {
            "ndcg": train_ndcg,
            "recall": {f"Recall@{k}": train_recall[k] for k in k_values}
        },
        "validation": {
            "ndcg": val_ndcg,
            "recall": {f"Recall@{k}": val_recall[k] for k in k_values}
        }
    }
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Model statistics saved to {stats_file}")

elif args.search_mode == "random":
    print("Random search enabled. Splitting train/validation...")
    unique_qids = np.unique(qids)
    train_qids, val_qids = train_test_split(unique_qids, test_size=0.2, random_state=42)
    train_idx = np.isin(qids, train_qids)
    val_idx = np.isin(qids, val_qids)

    train_data, train_labels, train_qids_arr = data[train_idx], labels[train_idx], qids[train_idx]
    val_data, val_labels, val_qids_arr = data[val_idx], labels[val_idx], qids[val_idx]

    train_groups = group_queries(train_qids_arr)
    val_groups = group_queries(val_qids_arr)

    dtrain = xgb.DMatrix(train_data, label=train_labels)
    dtrain.set_group(train_groups)
    dval = xgb.DMatrix(val_data, label=val_labels)
    dval.set_group(val_groups)

    param_space = {
        "eta": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 6, 8, 10],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5]
    }
    num_trials = 20
    num_rounds = 100
    best_ndcg = -1
    best_params = None
    best_model = None

    for i in range(num_trials):
        rand_params = {k: random.choice(v) for k, v in param_space.items()}
        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg",
            **rand_params
        }
        bst = xgb.train(params, dtrain, num_boost_round=num_rounds)
        val_preds = bst.predict(dval)
        val_ndcg, _ = calculate_metrics(val_labels, val_preds, val_qids_arr, k_values)
        print(f"Trial {i+1}/{num_trials} Params: {rand_params}, Validation NDCG: {val_ndcg}")
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_params = rand_params
            best_model = bst

    print(f"Best Params: {best_params}, Best Validation NDCG: {best_ndcg}")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    best_model.save_model(model_file)
    print(f"Best model saved to {model_file}")

    # Evaluate on train and val
    train_preds = best_model.predict(dtrain)
    train_ndcg, train_recall = calculate_metrics(train_labels, train_preds, train_qids_arr, k_values)
    val_preds = best_model.predict(dval)
    val_ndcg, val_recall = calculate_metrics(val_labels, val_preds, val_qids_arr, k_values)

    stats_file = f"{args.dir}/model_stats_randomsearch.json"
    stats = {
        "best_params": best_params,
        "train": {
            "ndcg": train_ndcg,
            "recall": {f"Recall@{k}": train_recall[k] for k in k_values}
        },
        "validation": {
            "ndcg": val_ndcg,
            "recall": {f"Recall@{k}": val_recall[k] for k in k_values}
        }
    }
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Model statistics saved to {stats_file}")

else:
    groups = group_queries(qids)
    dtrain = xgb.DMatrix(data, label=labels)
    dtrain.set_group(groups)
    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "eta": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    print("Evaluating before training...")
    initial_preds = np.zeros_like(labels)
    initial_ndcg, initial_recall = calculate_metrics(labels, initial_preds, qids, k_values)
    print(f"Initial NDCG: {initial_ndcg}")
    for k in k_values:
        print(f"Initial Recall@{k}: {initial_recall[k]}")
    print("\nTraining LambdaMART model...")
    num_rounds = 100
    bst = xgb.train(params, dtrain, num_boost_round=num_rounds)
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    bst.save_model(model_file)
    print(f"Model saved to {model_file}")
    print("\nEvaluating after training...")
    final_preds = bst.predict(dtrain)
    final_ndcg, final_recall = calculate_metrics(labels, final_preds, qids, k_values)
    print(f"Final NDCG: {final_ndcg}")
    for k in k_values:
        print(f"Final Recall@{k}: {final_recall[k]}")
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