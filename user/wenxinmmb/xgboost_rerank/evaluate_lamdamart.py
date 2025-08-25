import argparse
import xgboost as xgb
from sklearn.metrics import ndcg_score
import numpy as np
from collections import defaultdict

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

def load_dataset(file_path, qrel):
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
            
            # Get the label from the QREL file
            qid = int(main_parts[1].split(':')[1])
            label = qrel[qid].get(doc_id, 0)  # Default to 0 if not in QREL
            labels.append(label)
    
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

def evaluate_model(model_path, dataset_path, qrel_path, baseline_run_path, output_path):
    """Evaluate LambdaMART model and compare metrics."""
    # Load QREL file
    qrel = load_qrel(qrel_path)
    
    # Load test dataset
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
    save_trec_format(output_path, reranked_results)
    
    # Compute metrics for reranked results
    k_values = [10, 100, 1000]
    avg_ndcg, avg_recall = compute_metrics(labels, predictions, queries, k_values)
    for k, ndcg in avg_ndcg.items():
        print(f"Reranked Results - NDCG@{k}: {ndcg:.4f}")
    for k, recall in avg_recall.items():
        print(f"Reranked Results - Recall@{k}: {recall:.4f}")
    print('-'*20)
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
    
    avg_ndcg, avg_recall = compute_metrics(baseline_labels, baseline_predictions, baseline_queries, k_values)
    for k, ndcg in avg_ndcg.items():
        print(f"Baseline Results - NDCG@{k}: {ndcg:.4f}")
    for k, recall in avg_recall.items():
        print(f"Baseline Results - Recall@{k}: {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LambdaMART model and compare metrics.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LambdaMART model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--qrel_path", type=str, required=True, help="Path to the QREL file.")
    parser.add_argument("--baseline_run_path", type=str, required=True, help="Path to the baseline retrieval results.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the reranked results in TREC format.")
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.dataset_path, args.qrel_path, args.baseline_run_path, args.output_path)