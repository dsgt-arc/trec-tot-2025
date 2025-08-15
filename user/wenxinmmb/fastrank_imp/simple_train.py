#!/usr/bin/env python3
"""
Simple fastrank v0.8.0 training script for ToT dataset
This is a minimal version to get basic functionality working.
"""

import json
from xml.parsers.expat import model
import numpy as np
from pathlib import Path
from collections import defaultdict
import math
from typing import Dict, List, Tuple
import os

from fastrank import CModel, CDataset, TrainRequest


def load_queries(queries_file: str) -> Dict[str, str]:
    """Load queries from JSONL file"""
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            queries[data['query_id']] = data['query']
    return queries


def load_qrels(qrel_file: str) -> Dict[str, Dict[str, int]]:
    """Load relevance judgments from qrel file"""
    qrels = defaultdict(dict)
    with open(qrel_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels[qid][docid] = rel
    return dict(qrels)


def load_run_results(run_file: str) -> Dict[str, List[Tuple[str, float, int]]]:
    """Load BM25 run results"""
    results = defaultdict(list)
    with open(run_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts
                results[qid].append((docid, float(score), int(rank)))
    
    # Sort by rank for each query
    for qid in results:
        results[qid].sort(key=lambda x: x[2])
    
    return dict(results)


def create_basic_features(bm25_score: float, rank: int, all_scores: List[float]) -> List[float]:
    """Create basic ranking features"""
    features = []
    
    # BM25 score features
    features.extend([
        bm25_score,                    # Raw BM25 score
        math.log1p(bm25_score),       # Log BM25 score
        math.sqrt(bm25_score),        # Sqrt BM25 score
    ])
    
    # Rank features
    features.extend([
        rank,                         # Rank position
        1.0 / (1.0 + rank),          # Inverse rank
        math.log1p(rank),            # Log rank
    ])
    
    # Score normalization
    if len(all_scores) > 1:
        max_score = max(all_scores)
        min_score = min(all_scores)
        features.extend([
            bm25_score / max_score,                                          # Normalized by max
            (bm25_score - min_score) / (max_score - min_score + 1e-6),     # MinMax normalization
        ])
    else:
        features.extend([1.0, 1.0])
    
    return features


def create_training_data(queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], 
                        run_results: Dict[str, List[Tuple[str, float, int]]], 
                        output_file: str) -> None:
    """Create training data in RankLib format"""
    
    sample_count = 0
    
    with open(output_file, 'w') as f:
        for qid in queries:
            if qid not in run_results:
                continue
            
            query_qrels = qrels.get(qid, {})
            query_results = run_results[qid]
            
            # Extract all scores for normalization features
            all_scores = [score for _, score, _ in query_results]
            
            for doc_id, bm25_score, rank in query_results:
                # Get relevance label (default 0 if not in qrels)
                relevance = query_qrels.get(doc_id, 0)
                
                # Extract features
                features = create_basic_features(bm25_score, rank, all_scores)
                
                # Write in RankLib format: relevance qid:query_id feature_vector
                feature_str = " ".join([f"{i+1}:{feat}" for i, feat in enumerate(features)])
                f.write(f"{relevance} qid:{qid} {feature_str}\n")
                sample_count += 1
    
    print(f"Created {output_file} with {sample_count} samples and {len(features)} features")


def main():
    # Data paths (adjust as needed)
    data_dir = "/home/wenxin/project/data/2025"
    baseline_dir = "../baseline2024"
    
    train_queries_file = f"{data_dir}/generated-queries/llm-set1/train/queries.jsonl"
    train_qrel_file = f"{data_dir}/generated-queries/llm-set1/train/qrel.txt"
    train_run_file = f"{baseline_dir}/pyterrier-bm25-retrieval/runs/set1-train.txt"
    
    dev_queries_file = f"{data_dir}/generated-queries/llm-set1/dev/queries.jsonl"
    dev_qrel_file = f"{data_dir}/generated-queries/llm-set1/dev/qrel.txt"
    dev_run_file = f"{baseline_dir}/pyterrier-bm25-retrieval/runs/set1-dev.txt"
    
    # Load data
    print("Loading training data...")
    train_queries = load_queries(train_queries_file)
    train_qrels = load_qrels(train_qrel_file)
    train_results = load_run_results(train_run_file)
    
    print("Loading development data...")
    dev_queries = load_queries(dev_queries_file)
    dev_qrels = load_qrels(dev_qrel_file)
    dev_results = load_run_results(dev_run_file)
    
    print(f"Loaded {len(train_queries)} training queries, {len(dev_queries)} dev queries")
    
    # Create training files
    print("Creating training data files...")
    train_file = "tot_train_simple.txt"
    dev_file = "tot_dev_simple.txt"
    
    # Check if training files already exist, if not create them
    if not os.path.exists(train_file):
        create_training_data(train_queries, train_qrels, train_results, train_file)
    else:
        print(f"Training file {train_file} already exists, skipping creation")
        
    if not os.path.exists(dev_file):
        create_training_data(dev_queries, dev_qrels, dev_results, dev_file)
    else:
        print(f"Development file {dev_file} already exists, skipping creation")

    # Train model
    print("Training model...")
    
    # Load datasets using the correct API - open_ranksvm for RankLib format
    print(f"Loading training dataset: {train_file}")
    train_dataset = CDataset.open_ranksvm(train_file)
    print(f"Loading development dataset: {dev_file}")
    dev_dataset = CDataset.open_ranksvm(dev_file)
    print("Successfully loaded datasets with open_ranksvm method")
    
    print(f"Training dataset: {train_dataset.num_instances()} instances, {train_dataset.num_features()} features")
    print(f"Development dataset: {dev_dataset.num_instances()} instances, {dev_dataset.num_features()} features")
    
    # Train model using the dataset's train_model method with TrainRequest
    print("Training CoordinateAscent model...")
    
    # Create training request and configure for coordinate ascent
    train_request = TrainRequest()
    train_request = train_request.coordinate_ascent()
    print("Successfully configured TrainRequest for coordinate_ascent")
    print(f"TrainRequest params: {train_request.params if hasattr(train_request, 'params') else 'No params attribute'}")
    
    # Train the model with the TrainRequest
    print("Starting model training...")
    model = train_dataset.train_model(train_request)
    print("Model training successful!")
    
    # Save model
    model_file = "tot_simple_model.json"
    model_dict = model.to_dict()
    with open(model_file, 'w') as f:
        json.dump(model_dict, f, indent=2)
    print(f"Model saved to: {model_file}")
    
    # Test prediction and evaluation on development set
    print("Testing prediction on development set...")
    predictions = dev_dataset.predict_scores(model)
    print(f"Prediction successful! Got {len(predictions)} predictions")
    
    # Calculate some basic metrics
    if isinstance(predictions, list) and len(predictions) > 10:
        print(f"Sample predictions: {predictions[:10]}")
    else:
        print(f"First few predictions: {list(predictions)[:10] if hasattr(predictions, '__iter__') else predictions}")
    
    # Skip the dataset evaluate method since it requires an evaluator parameter
    # Use the separate evaluation script instead
    print("\nFor proper evaluation with NDCG and Recall metrics, run:")
    print("python evaluate_model.py")
    
    # Manual evaluation - calculate simple metrics
    print("Basic prediction statistics:")
    print(f"  Min prediction: {min(predictions) if predictions else 'N/A'}")
    print(f"  Max prediction: {max(predictions) if predictions else 'N/A'}")
    print(f"  Mean prediction: {sum(predictions)/len(predictions) if predictions else 'N/A'}")


if __name__ == "__main__":
    main()
