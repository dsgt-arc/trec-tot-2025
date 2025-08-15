#!/usr/bin/env python3
"""
Proper evaluation of the trained CoordinateAscent model
This script calculates NDCG, precision, and other ranking metrics
"""

import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple
from fastrank import CDataset, CModel

def load_qrel_data(qrel_file: str) -> Dict[str, Dict[str, int]]:
    """Load qrel data for evaluation"""
    qrels = defaultdict(dict)
    with open(qrel_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels[qid][docid] = rel
    return dict(qrels)

def load_trec_run(run_file: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load TREC run format results"""
    run_results = defaultdict(list)
    with open(run_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts
                run_results[qid].append((docid, float(score)))
    
    # Sort by score descending for each query
    for qid in run_results:
        run_results[qid].sort(key=lambda x: x[1], reverse=True)
    
    return dict(run_results)

def evaluate_trec_run(run_results: Dict[str, List[Tuple[str, float]]], qrel_file: str) -> Dict[str, float]:
    """Evaluate TREC run results against qrels"""
    qrels = load_qrel_data(qrel_file)
    
    metrics = {
        'ndcg@10': 0.0,
        'ndcg@100': 0.0,
        'recall@10': 0.0,
        'recall@100': 0.0,
        'num_queries': 0
    }
    
    for qid, doc_scores in run_results.items():
        if qid not in qrels:
            continue
            
        metrics['num_queries'] += 1
        query_qrels = qrels[qid]
        
        # Get relevance scores in ranked order
        relevances = []
        for docid, _ in doc_scores:
            relevances.append(query_qrels.get(docid, 0))
        
        # Calculate metrics
        metrics['ndcg@10'] += calculate_ndcg(relevances, 10)
        metrics['ndcg@100'] += calculate_ndcg(relevances, 100)
        metrics['recall@10'] += calculate_recall_at_k(relevances, 10)
        metrics['recall@100'] += calculate_recall_at_k(relevances, 100)
    
    # Average metrics
    num_queries = metrics['num_queries']
    if num_queries > 0:
        for metric in ['ndcg@10', 'ndcg@100', 'recall@10', 'recall@100']:
            metrics[metric] /= num_queries
    
    return metrics

def calculate_dcg(relevances: List[int], k: int = 10) -> float:
    """Calculate Discounted Cumulative Gain"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg

def calculate_ndcg(relevances: List[int], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain"""
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0

def calculate_recall_at_k(relevances: List[int], k: int = 10) -> float:
    """Calculate Recall@K"""
    total_relevant = sum(1 for rel in relevances if rel > 0)
    if total_relevant == 0:
        return 0.0
    relevant_at_k = sum(1 for rel in relevances[:k] if rel > 0)
    return relevant_at_k / total_relevant

def evaluate_predictions(predictions: List[float], ranklib_file: str, qrel_file: str) -> Dict[str, float]:
    """
    Evaluate model predictions against ground truth
    """
    # Load ground truth data
    qrels = load_qrel_data(qrel_file)
    
    # Group predictions by query
    prediction_idx = 0
    query_predictions = defaultdict(list)
    
    with open(ranklib_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 2:
                relevance = int(parts[0])
                qid = parts[1].split(':')[1]
                pred_score = predictions[prediction_idx] if prediction_idx < len(predictions) else 0.0
                query_predictions[qid].append((relevance, pred_score))
                prediction_idx += 1
    
    # Calculate metrics for each query
    metrics = {
        'ndcg@10': 0.0,
        'ndcg@100': 0.0,
        'recall@10': 0.0,
        'recall@100': 0.0,
        'num_queries': len(query_predictions)
    }
    
    for qid, preds in query_predictions.items():
        # Sort by predicted score (descending)
        preds.sort(key=lambda x: x[1], reverse=True)
        
        # Extract relevance scores in predicted order
        relevances = [rel for rel, _ in preds]
        
        # Calculate metrics
        metrics['ndcg@10'] += calculate_ndcg(relevances, 10)
        metrics['ndcg@100'] += calculate_ndcg(relevances, 100)
        metrics['recall@10'] += calculate_recall_at_k(relevances, 10)
        metrics['recall@100'] += calculate_recall_at_k(relevances, 100)
    
    # Average metrics
    num_queries = metrics['num_queries']
    for metric in ['ndcg@10', 'ndcg@100', 'recall@10', 'recall@100']:
        metrics[metric] /= num_queries
    
    return metrics

def main():
    """Evaluate the trained model"""
    print("=== CoordinateAscent Model Evaluation ===")
    
    # Load the saved model
    model_file = "tot_simple_model.json"
    try:
        with open(model_file, 'r') as f:
            model_dict = json.load(f)
        
        # Recreate model from dictionary
        model = CModel.from_dict(model_dict)
        print(f"✓ Loaded model from {model_file}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load development dataset
    dev_file = "tot_dev_simple.txt"
    try:
        dev_dataset = CDataset.open_ranksvm(dev_file)
        print(f"✓ Loaded development dataset: {dev_dataset.num_instances()} instances")
    except Exception as e:
        print(f"Error loading development dataset: {e}")
        return
    
    # Make predictions
    try:
        predictions = dev_dataset.predict_scores(model)
        print(f"✓ Generated {len(predictions)} predictions")
    except Exception as e:
        print(f"Error making predictions: {e}")
        return
    
    # Evaluate predictions
    try:
        qrel_file = "/home/wenxin/project/data/2025/generated-queries/llm-set1/dev/qrel.txt"
        
        # Evaluate reranker
        reranker_metrics = evaluate_predictions(predictions, dev_file, qrel_file)
        
        # Evaluate baseline BM25
        baseline_run_file = "../baseline2024/pyterrier-bm25-retrieval/runs/set1-dev.txt"
        print(f"Loading baseline results from: {baseline_run_file}")
        
        try:
            baseline_results = load_trec_run(baseline_run_file)
            baseline_metrics = evaluate_trec_run(baseline_results, qrel_file)
            print(f"✓ Loaded baseline with {baseline_metrics['num_queries']} queries")
        except Exception as baseline_error:
            print(f"Error loading baseline: {baseline_error}")
            baseline_metrics = None
        
        # Display results
        print("\n=== EVALUATION RESULTS ===")
        print(f"Number of queries: {reranker_metrics['num_queries']}")
        
        print(f"\n--- CoordinateAscent Reranker ---")
        print(f"NDCG@10:          {reranker_metrics['ndcg@10']:.4f}")
        print(f"NDCG@100:         {reranker_metrics['ndcg@100']:.4f}")
        print(f"Recall@10:        {reranker_metrics['recall@10']:.4f}")
        print(f"Recall@100:       {reranker_metrics['recall@100']:.4f}")
        
        if baseline_metrics:
            print(f"\n--- BM25 Baseline ---")
            print(f"NDCG@10:          {baseline_metrics['ndcg@10']:.4f}")
            print(f"NDCG@100:         {baseline_metrics['ndcg@100']:.4f}")
            print(f"Recall@10:        {baseline_metrics['recall@10']:.4f}")
            print(f"Recall@100:       {baseline_metrics['recall@100']:.4f}")
            
            print(f"\n=== PERFORMANCE COMPARISON ===")
            ndcg10_improvement = ((reranker_metrics['ndcg@10'] / baseline_metrics['ndcg@10']) - 1) * 100 if baseline_metrics['ndcg@10'] > 0 else 0
            ndcg100_improvement = ((reranker_metrics['ndcg@100'] / baseline_metrics['ndcg@100']) - 1) * 100 if baseline_metrics['ndcg@100'] > 0 else 0
            recall10_improvement = ((reranker_metrics['recall@10'] / baseline_metrics['recall@10']) - 1) * 100 if baseline_metrics['recall@10'] > 0 else 0
            recall100_improvement = ((reranker_metrics['recall@100'] / baseline_metrics['recall@100']) - 1) * 100 if baseline_metrics['recall@100'] > 0 else 0
            
            print(f"NDCG@10 improvement:    {ndcg10_improvement:+.1f}%")
            print(f"NDCG@100 improvement:   {ndcg100_improvement:+.1f}%")
            print(f"Recall@10 improvement:  {recall10_improvement:+.1f}%")
            print(f"Recall@100 improvement: {recall100_improvement:+.1f}%")
            
            if ndcg10_improvement > 0:
                print(f"🎉 Your reranker IMPROVES over BM25 baseline!")
            else:
                print(f"⚠️  Your reranker performs worse than BM25 baseline")
        else:
            print(f"\n=== PERFORMANCE COMPARISON ===")
            print(f"Your CoordinateAscent NDCG@10: {reranker_metrics['ndcg@10']:.4f}")
            print(f"Baseline comparison not available")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
