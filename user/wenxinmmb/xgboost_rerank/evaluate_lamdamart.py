import argparse
import xgboost as xgb
from sklearn.metrics import ndcg_score
import numpy as np
from collections import defaultdict
import json
import os

def load_qrel(qrel_path):
    """Load relevance labels from a QREL file."""
    qrel = defaultdict(dict)
    with open(qrel_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid = int(parts[0])  # Query ID
            doc_id = parts[2]    # Document ID
            relevance = int(parts[3])  # Relevance label
            qrel[qid][doc_id] = relevance
    return qrel

def load_dataset(file_path, qrel=None):
    """Load dataset in LambdaMART format."""
    queries = []
    features = []
    doc_ids = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Split the line at the '#' symbol to separate the doc_id
            parts = line.strip().split('#')
            main_parts = parts[0].strip().split()  # The part before '#'
            doc_id = parts[1].strip() if len(parts) > 1 else None  # The doc_id
            
            queries.append(int(main_parts[1].split(':')[1]))
            feature_vector = [float(p.split(':')[1]) for p in main_parts[2:]]
            features.append(feature_vector)
            doc_ids.append(doc_id)
            
            # Get the label from the QREL file (skip if qrel is None)
            if qrel is not None:
                qid = int(main_parts[1].split(':')[1])
                label = qrel[qid].get(doc_id, 0)  # Default to 0 if not in QREL
                labels.append(label)
            else:
                labels.append(0)  # Default label when skipping evaluation
    
    return np.array(labels), np.array(queries), np.array(features), doc_ids

def load_baseline_run(file_path):
    """Load the baseline retrieval results."""
    baseline = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid = int(parts[0])
            doc_id = parts[2]
            score = float(parts[4])
            baseline[qid].append((doc_id, score))
    return baseline

def save_trec_format(output_path, reranked_results):
    """Save reranked results in TREC format."""
    with open(output_path, 'w') as f:
        for qid, docs in reranked_results.items():
            for rank, (doc_id, score) in enumerate(docs, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score} rerank\n")

def compute_metrics(labels, predictions, queries, k_values):
    """Compute NDCG and recall metrics."""
    unique_queries = np.unique(queries)
    ndcg_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    
    for qid in unique_queries:
        query_mask = (queries == qid)
        true_labels = labels[query_mask]
        pred_scores = predictions[query_mask]
        
        # Compute NDCG@k for each cutoff
        for k in k_values:
            ndcg = ndcg_score([true_labels], [pred_scores], k=k)
            ndcg_scores[k].append(ndcg)
        
        # Compute recall@k
        sorted_indices = np.argsort(-pred_scores)
        relevant_docs = np.sum(true_labels > 0)
        for k in k_values:
            top_k_relevant = np.sum(true_labels[sorted_indices[:k]] > 0)
            recall = top_k_relevant / relevant_docs if relevant_docs > 0 else 0
            recall_scores[k].append(recall)
    
    avg_ndcg = {k: np.mean(ndcg_scores[k]) for k in k_values}
    avg_recall = {k: np.mean(recall_scores[k]) for k in k_values}
    return avg_ndcg, avg_recall

def generate_rerank_results_only(model_path, output_dir):
    """Generate reranking results without evaluation."""
    dataset_path = os.path.join(output_dir, "features.txt")
    labels, queries, features, doc_ids = load_dataset(dataset_path, qrel=None)
    
    # Load the trained model
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Predict scores
    dtest = xgb.DMatrix(features)
    predictions = model.predict(dtest)
    
    # Group predictions by query
    reranked_results = defaultdict(list)
    for qid, doc_id, score in zip(queries, doc_ids, predictions):
        reranked_results[qid].append((doc_id, score))
    
    # Sort results by score
    for qid in reranked_results:
        reranked_results[qid].sort(key=lambda x: x[1], reverse=True)
    
    # Save reranked results in TREC format
    rerank_results_path = os.path.join(output_dir, f"rerank-results.txt")
    save_trec_format(rerank_results_path, reranked_results)
    print(f"Rerank results saved to {rerank_results_path}")

def evaluate_model(model_path, qrel_path, baseline_run_path, output_dir):
    """Evaluate LambdaMART model and compare metrics."""
    # Load QREL file
    qrel = load_qrel(qrel_path)
    dataset_path = os.path.join(output_dir, "features.txt")
    labels, queries, features, doc_ids = load_dataset(dataset_path, qrel)
    
    # Load the trained model
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Predict scores
    dtest = xgb.DMatrix(features)
    predictions = model.predict(dtest)
    
    # Group predictions by query
    reranked_results = defaultdict(list)
    for qid, doc_id, score in zip(queries, doc_ids, predictions):
        reranked_results[qid].append((doc_id, score))
    
    # Sort results by score
    for qid in reranked_results:
        reranked_results[qid].sort(key=lambda x: x[1], reverse=True)
    
    # Save reranked results in TREC format
    rerank_results_path = os.path.join(output_dir, f"rerank-results.txt")
    save_trec_format(rerank_results_path, reranked_results)
    print(f"Rerank results saved to {rerank_results_path}")
    
    # Compute metrics for reranked results
    k_values = [10, 100, 1000]
    avg_ndcg_reranked, avg_recall_reranked = compute_metrics(labels, predictions, queries, k_values)
    
    # Load baseline results and compute metrics
    baseline = load_baseline_run(baseline_run_path)
    baseline_labels = []
    baseline_predictions = []
    baseline_queries = []
    
    for qid, docs in baseline.items():
        for doc_id, score in docs:
            baseline_queries.append(qid)
            baseline_predictions.append(score)
            # Get the label from the QREL file
            label = qrel[qid].get(doc_id, 0)  # Default to 0 if not in QREL
            baseline_labels.append(label)
    
    baseline_labels = np.array(baseline_labels)
    baseline_predictions = np.array(baseline_predictions)
    baseline_queries = np.array(baseline_queries)
    
    avg_ndcg_baseline, avg_recall_baseline = compute_metrics(baseline_labels, baseline_predictions, baseline_queries, k_values)
    
    # Prepare results for JSON output
    results = {
        "Baseline Results": {
            "NDCG": {k: round(avg_ndcg_baseline[k], 2) for k in k_values},
            "Recall": {k: round(avg_recall_baseline[k], 2) for k in k_values}
        },
        "Reranked Results": {
            "NDCG": {k: round(avg_ndcg_reranked[k], 2) for k in k_values},
            "Recall": {k: round(avg_recall_reranked[k], 2) for k in k_values}
        },
        "Changes": {
            "NDCG": {
                k: {
                    "Absolute Change": round(avg_ndcg_reranked[k] - avg_ndcg_baseline[k], 2),
                    "Percentage Change": round(((avg_ndcg_reranked[k] - avg_ndcg_baseline[k]) / avg_ndcg_baseline[k] * 100), 2) if avg_ndcg_baseline[k] != 0 else 0
                }
                for k in k_values
            },
            "Recall": {
                k: {
                    "Absolute Change": round(avg_recall_reranked[k] - avg_recall_baseline[k], 2),
                    "Percentage Change": round(((avg_recall_reranked[k] - avg_recall_baseline[k]) / avg_recall_baseline[k] * 100), 2) if avg_recall_baseline[k] != 0 else 0
                }
                for k in k_values
            }
        }
    }
    
    # Write results to JSON file
    stats_path = os.path.join(output_dir, f"rerank-stats.json")
    with open(stats_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Metrics and comparison saved to {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LambdaMART model and compare metrics.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LambdaMART model.")
    parser.add_argument("--qrel_path", type=str, help="Path to the QREL file.")
    parser.add_argument("--baseline_run_path", type=str, help="Path to the baseline retrieval results.")
    parser.add_argument("--dir", type=str, required=True, help="Directory to look up dataset and save the results.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation and only generate reranking results.")
    
    args = parser.parse_args()
    
    if args.skip_eval:
        # Generate reranking results without evaluation
        generate_rerank_results_only(args.model_path, args.dir)
    else:
        # Require qrel_path and baseline_run_path for evaluation
        if not args.qrel_path or not args.baseline_run_path:
            parser.error("--qrel_path and --baseline_run_path are required when not using --skip_eval")
        evaluate_model(args.model_path, args.qrel_path, args.baseline_run_path, args.dir)