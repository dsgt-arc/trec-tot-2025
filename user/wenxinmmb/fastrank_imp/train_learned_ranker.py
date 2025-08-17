import argparse
import os
import sys
import json
from collections import defaultdict
import numpy as np
from fastrank import CModel, CDataset, TrainRequest
from select_train_samples import extract_features_v2, load_retrieval_results

def load_training_data(data_file):
    """Load training data in RankLib format."""
    queries = defaultdict(list)
    
    with open(data_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split('#')
            ranklib_part = parts[0].strip()
            doc_id = parts[1].strip() if len(parts) > 1 else ""
            
            # Parse RankLib format: label qid:query_id 1:feat1 2:feat2 ...
            tokens = ranklib_part.split()
            label = int(tokens[0])
            
            qid_token = tokens[1]
            query_id = qid_token.split(':')[1]
            
            # Extract features
            features = []
            for token in tokens[2:]:
                feat_val = float(token.split(':')[1])
                features.append(feat_val)
            
            queries[query_id].append({
                'label': label,
                'features': features,
                'doc_id': doc_id
            })
    
    return queries

def load_qrels(qrel_file):
    """Load qrels file and return dict of query_id -> set of relevant doc_ids."""
    qrels = {}
    with open(qrel_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_id = parts[0]
            doc_id = parts[2]
            relevance = int(parts[3])
            
            if relevance > 0:
                if query_id not in qrels:
                    qrels[query_id] = set()
                qrels[query_id].add(doc_id)
    return qrels

def get_retrieval_pairs_from_file(file_path):
    """Load retrieval results from a file."""
    retrieval_pairs = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts= line.strip().split()
            query_id = parts[0]
            doc_id = parts[2]
            retrieval_pairs[query_id].append(doc_id)
    return retrieval_pairs

def calculate_dcg(relevance_scores, k=None):
    """Calculate Discounted Cumulative Gain."""
    if k is not None:
        relevance_scores = relevance_scores[:k]
    
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += (2**rel - 1) / np.log2(i + 2)
    return dcg

def calculate_ndcg(ranked_docs, qrels, query_id, k=None):
    """Calculate Normalized Discounted Cumulative Gain at k."""
    if query_id not in qrels:
        return 0.0
    
    relevant_docs = qrels[query_id]
    
    # Get relevance scores for ranked documents
    relevance_scores = []
    for doc in ranked_docs:
        if k is not None and len(relevance_scores) >= k:
            break
        relevance_scores.append(1 if doc['doc_id'] in relevant_docs else 0)
    
    # Calculate DCG
    dcg = calculate_dcg(relevance_scores, k)
    
    # Calculate IDCG (ideal DCG)
    ideal_scores = [1] * len(relevant_docs) + [0] * max(0, (k or len(ranked_docs)) - len(relevant_docs))
    if k is not None:
        ideal_scores = ideal_scores[:k]
    idcg = calculate_dcg(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def calculate_recall(ranked_docs, qrels, query_id, k=None):
    """Calculate Recall@k."""
    if query_id not in qrels:
        return 0.0
    
    relevant_docs = qrels[query_id]
    total_relevant = len(relevant_docs)
    
    if total_relevant == 0:
        return 0.0
    
    # Get top-k documents
    top_k_docs = ranked_docs[:k] if k is not None else ranked_docs
    
    # Count relevant documents in top-k
    relevant_in_top_k = 0
    for doc in top_k_docs:
        if doc['doc_id'] in relevant_docs:
            relevant_in_top_k += 1
    
    return relevant_in_top_k / total_relevant

def train_coordinate_ascent(training_file):
    """Train coordinate ascent ranker using fastrank CDataset and TrainRequest."""
    
    print(f"Training coordinate ascent ranker from file: {training_file}")
    
    # Load dataset using the correct API
    train_dataset = CDataset.open_ranksvm(training_file)
    
    print(f"Training dataset: {train_dataset.num_instances()} instances, {train_dataset.num_features()} features")
    
    # Create training request for coordinate ascent
    train_request = TrainRequest()
    train_request = train_request.coordinate_ascent()
    
    # Train the model
    print("Starting model training...")
    ranker = train_dataset.train_model(train_request)
    print("Model training successful!")
    
    return ranker

def analyze_training_data(training_file):
    """Analyze training data quality for debugging."""
    print("=== TRAINING DATA ANALYSIS ===")
    
    queries = defaultdict(list)
    total_samples = 0
    pos_samples = 0
    neg_samples = 0
    feature_values = defaultdict(list)
    
    with open(training_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('#') or not line.strip():
                continue
                
            try:
                parts = line.strip().split('#')
                ranklib_part = parts[0].strip()
                tokens = ranklib_part.split()
                
                label = int(tokens[0])
                qid = tokens[1].split(':')[1]
                
                # Count samples
                total_samples += 1
                if label > 0:
                    pos_samples += 1
                else:
                    neg_samples += 1
                
                queries[qid].append(label)
                
                # Analyze features
                for token in tokens[2:]:
                    feat_idx, feat_val = token.split(':')
                    feature_values[int(feat_idx)].append(float(feat_val))
                    
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                print(f"Line content: {line.strip()}")
    
    print(f"Total samples: {total_samples}")
    print(f"Positive samples: {pos_samples} ({pos_samples/total_samples*100:.1f}%)")
    print(f"Negative samples: {neg_samples} ({neg_samples/total_samples*100:.1f}%)")
    print(f"Unique queries: {len(queries)}")
    print(f"Avg samples per query: {total_samples/len(queries):.1f}")
    
    # Query-level analysis
    pos_per_query = [sum(1 for label in labels if label > 0) for labels in queries.values()]
    neg_per_query = [sum(1 for label in labels if label == 0) for labels in queries.values()]
    
    print(f"Positive docs per query: min={min(pos_per_query)}, max={max(pos_per_query)}, avg={np.mean(pos_per_query):.1f}")
    print(f"Negative docs per query: min={min(neg_per_query)}, max={max(neg_per_query)}, avg={np.mean(neg_per_query):.1f}")
    
    # Feature analysis
    print(f"Number of features: {len(feature_values)}")
    for feat_idx in sorted(feature_values.keys()):  # Show all features
        values = feature_values[feat_idx]
        print(f"Feature {feat_idx}: min={np.min(values):.4f}, max={np.max(values):.4f}, mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    
    # Check for potential issues
    zero_variance_features = []
    for feat_idx, values in feature_values.items():
        if np.std(values) < 1e-6:
            zero_variance_features.append(feat_idx)
    
    if zero_variance_features:
        print(f"WARNING: Found {len(zero_variance_features)} features with zero/low variance: {zero_variance_features[:10]}")
    
    return queries, feature_values

def evaluate_baseline(retrieval_results, qrels, k=10):
    """Evaluate baseline retrieval performance (no reranking)."""
    print("=== BASELINE EVALUATION ===")
    
    ndcg_scores = []
    recall_scores = []
    
    for query_id in retrieval_results:
        if query_id not in qrels:
            continue
            
        # Use original ranking (no reranking)
        ranked_docs = [{'doc_id': doc_id} for doc_id in retrieval_results[query_id]]
        
        # Calculate baseline metrics
        ndcg = calculate_ndcg(ranked_docs, qrels, query_id, k)
        recall = calculate_recall(ranked_docs, qrels, query_id, k)
        
        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
    
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    print(f"Baseline NDCG@{k}: {avg_ndcg:.4f}")
    print(f"Baseline Recall@{k}: {avg_recall:.4f}")
    print(f"Evaluated {len(ndcg_scores)} queries")
    
    return avg_ndcg, avg_recall

def load_model_from_json(model_path):
    """Load a trained model from JSON file."""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'r') as f:
        model_dict = json.load(f)
    
    # Create model from dictionary
    ranker = CModel.from_dict(model_dict)
    print("Model loaded successfully!")
    
    return ranker

def evaluate_reranking(ranker, retrieval_results, qrels, sparse_results, dense_results, k=10):
    """Evaluate reranking performance using NDCG@k and Recall@k."""
    
    ndcg_scores = []
    recall_scores = []
    
    print(f"Starting evaluation on {len(retrieval_results)} queries...")
    processed_queries = 0
    
    for query_id in retrieval_results:
        if query_id not in qrels:
            continue
        
        # Extract features for all documents in retrieval results
        docs_with_features = []
        for doc_id in retrieval_results[query_id]:
            features = extract_features_v2(doc_id, query_id, sparse_results, dense_results)
            docs_with_features.append({
                'doc_id': doc_id,
                'features': features,
            })
        
        if not docs_with_features:
            continue
        
        # Create temporary dataset for prediction
        temp_samples = []
        for i, doc in enumerate(docs_with_features):
            features_str = " ".join([f"{j+1}:{feat}" for j, feat in enumerate(doc['features'])])
            temp_samples.append(f"0 qid:{query_id} {features_str} # {doc['doc_id']}")
        # Write to temporary file and predict
        temp_file = f"temp_eval_{query_id}.txt"
        with open(temp_file, 'w') as f:
            for line in temp_samples:
                f.write(line + '\n')

        # Load dataset and predict
        eval_dataset = CDataset.open_ranksvm(temp_file)
        rerank_scores = eval_dataset.predict_scores(ranker)

        # Sort documents by reranking scores
        for i, doc in enumerate(docs_with_features):
            doc['rerank_score'] = rerank_scores[i]
        
        reranked_docs = sorted(docs_with_features, key=lambda x: x['rerank_score'], reverse=True)

        # Calculate NDCG@k and Recall@k
        ndcg = calculate_ndcg(reranked_docs, qrels, query_id, k)
        recall = calculate_recall(reranked_docs, qrels, query_id, k)
        
        ndcg_scores.append(ndcg)
        recall_scores.append(recall)

        processed_queries += 1
        if processed_queries % 50 == 0:
            print(f"Processed {processed_queries} queries...")

        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    avg_ndcg = np.mean(ndcg_scores)
    avg_recall = np.mean(recall_scores)

    print(f"Evaluation complete. Processed {processed_queries} queries with relevant documents.")
    if ndcg_scores:
        print(f"Score distribution - NDCG: min={np.min(ndcg_scores):.4f}, max={np.max(ndcg_scores):.4f}, std={np.std(ndcg_scores):.4f}")
        print(f"Score distribution - Recall: min={np.min(recall_scores):.4f}, max={np.max(recall_scores):.4f}, std={np.std(recall_scores):.4f}")

    return avg_ndcg, avg_recall

def main():
    parser = argparse.ArgumentParser(description='Train coordinate ascent reranker')
    
    # Data paths
    parser.add_argument('--training_data', default='outputs/training_dev_samples.txt',
                       help='Path to training data in RankLib format')
    parser.add_argument('--qrel_file', default='/home/wenxin/project/data/2025/generated-queries/llm-set1/dev/qrel.txt',
                       help='Path to qrel file for evaluation')
    parser.add_argument('--retrieval_results', default='inputs/llm-set1-combined.txt',
                       help='Path to first-level retrieval results')
    parser.add_argument('--sparse_results', default='inputs/llm-set1-bm25-run.txt',
                       help='Path to sparse retrieval results')
    parser.add_argument('--dense_results', default='inputs/llm-set1-bge-dense-run.txt',
                       help='Path to dense retrieval results')
    parser.add_argument('--model_output', default='outputs/models/feature_v2/model_parameters.json',
                       help='Path to save trained model')
    
    # Mode selection
    parser.add_argument('--eval_only', action='store_true',
                       help='Skip training and only run evaluation using existing model')
    parser.add_argument('--model_path', default=None,
                       help='Path to existing model JSON file for evaluation-only mode')
    
    # Evaluation parameters
    parser.add_argument('--eval_k', type=int, default=10,
                       help='k value for NDCG@k evaluation')
    
    # Debugging and analysis
    parser.add_argument('--analyze_data', action='store_true',
                       help='Analyze training data quality before training')
    parser.add_argument('--baseline_eval', action='store_true',
                        help='Evaluate baseline retrieval performance without reranking')
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Evaluation-only mode
        print("=== EVALUATION-ONLY MODE ===")
        
        # Determine model path
        model_path = args.model_path if args.model_path else args.model_output
        
        # Load existing model
        try:
            ranker = load_model_from_json(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
            
    else:
        # Training mode
        print("=== TRAINING MODE ===")
        
        # Verify training data file exists
        if not os.path.exists(args.training_data):
            print(f"Error: Training data file not found: {args.training_data}")
            return
        
        # Analyze training data if requested
        if args.analyze_data:
            analyze_training_data(args.training_data)
            print()
        
        # Train the ranker directly from the training file
        ranker = train_coordinate_ascent(args.training_data)
        
        # Save the model
        os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
        model_dict = ranker.to_dict()
        with open(args.model_output, 'w') as f:
            json.dump(model_dict, f, indent=2)
        print(f"Model saved as JSON to {args.model_output}")

    
    # Evaluate the ranker
    if args.qrel_file and args.retrieval_results:
        print("Evaluating reranking performance...")
        
        # Load evaluation data
        qrels = load_qrels(args.qrel_file)

        # Load sparse and dense results for feature extraction
        print("Loading sparse retrieval results...")
        sparse_results = load_retrieval_results(args.sparse_results)
        
        print("Loading dense retrieval results...")
        dense_results = load_retrieval_results(args.dense_results)

        validation_set_pairs = get_retrieval_pairs_from_file(args.retrieval_results)

        # Evaluate baseline if requested
        if args.baseline_eval:
            baseline_ndcg, baseline_recall = evaluate_baseline(validation_set_pairs, qrels, args.eval_k)
            print()

        # Calculate NDCG@k and Recall@k
        ndcg, recall = evaluate_reranking(ranker, validation_set_pairs, qrels, sparse_results, dense_results, args.eval_k)
        print(f"Reranked NDCG@{args.eval_k}: {ndcg:.4f}")
        print(f"Reranked Recall@{args.eval_k}: {recall:.4f}")
        
        # Show improvement if baseline was computed
        if args.baseline_eval:
            ndcg_improvement = ((ndcg - baseline_ndcg) / baseline_ndcg * 100) if baseline_ndcg > 0 else 0
            recall_improvement = ((recall - baseline_recall) / baseline_recall * 100) if baseline_recall > 0 else 0
            print(f"NDCG improvement: {ndcg_improvement:+.1f}%")
            print(f"Recall improvement: {recall_improvement:+.1f}%")
    else:
        print("Skipping evaluation - missing qrel file or retrieval results")

if __name__ == "__main__":
    main()
