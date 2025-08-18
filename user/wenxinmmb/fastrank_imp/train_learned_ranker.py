import argparse
import os
import sys
import json
from collections import defaultdict
import numpy as np
from fastrank import CModel, CDataset, TrainRequest
from select_train_samples import extract_features_v2, extract_features_v1, extract_features_v3, load_retrieval_results

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
    
    # Create training request for coordinate ascent with improved hyperparameters
    train_request = TrainRequest.coordinate_ascent()
    params = train_request.params
    
    # Key hyperparameters to address underfitting
    params.num_max_iterations = 100  # Increased from default 25 - allows more training time
    params.tolerance = 0.0001       # Decreased from default 0.001 - tighter convergence
    params.step_base = 0.1          # Increased from default 0.05 - larger learning steps
    params.num_restarts = 10        # Increased from default 5 - more chances to find global optimum
    params.output_ensemble = True  # Single model (could try True later)
    
    # Other good settings
    params.init_random = True       # Random initialization
    params.normalize = True         # Feature normalization
    params.step_scale = 2.0         # Default scaling factor
    params.seed = 1234567          # Fixed seed for reproducibility
    params.quiet = False           # Show training progress
    
    print("Training hyperparameters:")
    print(f"  Max iterations: {params.num_max_iterations}")
    print(f"  Tolerance: {params.tolerance}")
    print(f"  Step base: {params.step_base}")
    print(f"  Num restarts: {params.num_restarts}")
    
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

def evaluate_baseline(retrieval_results, qrels, k_values=[10, 100, 1000]):
    """Evaluate baseline retrieval performance (no reranking) at multiple k values."""
    print("=== BASELINE EVALUATION ===")
    
    results = {}
    for k in k_values:
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
        
        results[k] = {
            'ndcg': avg_ndcg,
            'recall': avg_recall,
            'num_queries': len(ndcg_scores)
        }
        
        print(f"Baseline NDCG@{k}: {avg_ndcg:.4f}")
        print(f"Baseline Recall@{k}: {avg_recall:.4f}")
    
    print(f"Evaluated {results[k_values[0]]['num_queries']} queries")
    return results

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

def evaluate_reranking(ranker, retrieval_results, qrels, sparse_results, dense_results, k_values=[10, 100, 1000], feature_version='v2'):
    """Evaluate reranking performance using NDCG@k and Recall@k at multiple k values."""
    
    # Select feature extraction function
    # Select appropriate feature extraction function
    if feature_version == 'v1':
        extract_features = extract_features_v1
    elif feature_version == 'v2':
        extract_features = extract_features_v2
    elif feature_version == 'v3':
        extract_features = extract_features_v3
    else:
        raise ValueError(f"Invalid feature_version: {feature_version}. Must be 'v1', 'v2', or 'v3'.")
    
    print(f"Starting evaluation on {len(retrieval_results)} queries using feature version {feature_version}...")
    processed_queries = 0
    
    # Initialize results storage
    results = {}
    for k in k_values:
        results[k] = {
            'ndcg_scores': [],
            'recall_scores': []
        }
    
    for query_id in retrieval_results:
        if query_id not in qrels:
            continue
        
        # Extract features for all documents in retrieval results
        docs_with_features = []
        for doc_id in retrieval_results[query_id]:
            features = extract_features(doc_id, query_id, sparse_results, dense_results)
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

        # Calculate NDCG@k and Recall@k for all k values
        for k in k_values:
            ndcg = calculate_ndcg(reranked_docs, qrels, query_id, k)
            recall = calculate_recall(reranked_docs, qrels, query_id, k)
            
            results[k]['ndcg_scores'].append(ndcg)
            results[k]['recall_scores'].append(recall)

        processed_queries += 1
        if processed_queries % 50 == 0:
            print(f"Processed {processed_queries} queries...")

        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Calculate averages for each k
    final_results = {}
    for k in k_values:
        ndcg_scores = results[k]['ndcg_scores']
        recall_scores = results[k]['recall_scores']
        
        final_results[k] = {
            'ndcg': np.mean(ndcg_scores),
            'recall': np.mean(recall_scores),
            'num_queries': len(ndcg_scores),
            'ndcg_std': np.std(ndcg_scores),
            'recall_std': np.std(recall_scores)
        }

    print(f"Evaluation complete. Processed {processed_queries} queries with relevant documents.")
    
    return final_results

def save_evaluation_results(baseline_results, reranked_results, model_version, feature_version, output_dir, split_name):
    """Save evaluation results to JSON file."""
    import datetime
    
    # Create evaluation results dictionary
    evaluation_data = {
        "metadata": {
            "model_version": model_version,
            "feature_version": feature_version,
            "evaluation_date": datetime.datetime.now().isoformat(),
            "k_values": [10, 100, 1000]
        },
        "baseline": baseline_results,
        "reranked": reranked_results,
        "improvements": {}
    }
    
    # Calculate improvements
    for k in [10, 100, 1000]:
        if k in baseline_results and k in reranked_results:
            baseline_ndcg = baseline_results[k]['ndcg']
            baseline_recall = baseline_results[k]['recall']
            reranked_ndcg = reranked_results[k]['ndcg']
            reranked_recall = reranked_results[k]['recall']
            
            ndcg_improvement = ((reranked_ndcg - baseline_ndcg) / baseline_ndcg * 100) if baseline_ndcg > 0 else 0
            recall_improvement = ((reranked_recall - baseline_recall) / baseline_recall * 100) if baseline_recall > 0 else 0
            
            evaluation_data["improvements"][k] = {
                "ndcg_improvement_percent": ndcg_improvement,
                "recall_improvement_percent": recall_improvement,
                "ndcg_absolute_improvement": reranked_ndcg - baseline_ndcg,
                "recall_absolute_improvement": reranked_recall - baseline_recall
            }
    
    # Save as JSON
    json_file = os.path.join(output_dir, f"evaluation_results_{split_name}.json")
    with open(json_file, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"Evaluation results saved to: {json_file}")
    
    return json_file

def main():
    parser = argparse.ArgumentParser(description='Train coordinate ascent reranker')
    
    # Model version argument
    parser.add_argument('--model_version', type=str, required=True,
                       help='Model version identifier (e.g., v2, v3, v4)')
    
    # Split argument
    parser.add_argument('--split', type=str, required=True,
                       help='Dataset split to use for evaluation (e.g., dev, train-100, train-500)')
    
    # Data paths
    parser.add_argument('--training_data', default=None,
                       help='Path to training data in RankLib format (will use model_version if not specified)')
    parser.add_argument('--qrel_file', default=None,
                       help='Path to qrel file for evaluation (will use split if not specified)')
    parser.add_argument('--retrieval_results', default='inputs/llm-set1-combined.txt',
                       help='Path to first-level retrieval results')
    parser.add_argument('--sparse_results', default='inputs/llm-set1-bm25-run.txt',
                       help='Path to sparse retrieval results')
    parser.add_argument('--dense_results', default='inputs/llm-set1-bge-dense-run.txt',
                       help='Path to dense retrieval results')
    parser.add_argument('--model_output', default=None,
                       help='Path to save trained model (will use model_version if not specified)')
    
    # Feature version selection
    parser.add_argument('--feature_version', choices=['v1', 'v2', 'v3'], required=True,
                       help='Feature version to use: v1 (original), v2 (normalized), or v3 (v1 + pageview + pagerank)')
    
    # Mode selection
    parser.add_argument('--eval_only', action='store_true',
                       help='Skip training and only run evaluation using existing model')
    parser.add_argument('--model_path', default=None,
                       help='Path to existing model JSON file for evaluation-only mode')
    
    # Debugging and analysis
    parser.add_argument('--analyze_data', action='store_true',
                       help='Analyze training data quality before training')
    
    args = parser.parse_args()
    
    # Set default paths based on model_version and split if not explicitly provided
    if args.training_data is None:
        args.training_data = f'outputs/models/model_{args.model_version}/training_data_{args.split}.txt'

    if args.qrel_file is None:
        args.qrel_file = f'/home/wenxin/project/data/2025/generated-queries/llm-set1/{args.split}/qrel.txt'
    
    if args.model_output is None:
        args.model_output = f'outputs/models/model_{args.model_version}/model_parameters.json'
    
    if args.eval_only:
        # Evaluation-only mode
        print("=== EVALUATION-ONLY MODE ===")
        print(f"Using feature version: {args.feature_version}")
        print(f"Using split: {args.split}")
        
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
        print(f"Using feature version: {args.feature_version}")
        print(f"Using split: {args.split}")
        
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

        # Evaluate baseline at multiple k values
        baseline_results = evaluate_baseline(validation_set_pairs, qrels)
        print()

        # Calculate NDCG@k and Recall@k at multiple k values
        reranked_results = evaluate_reranking(ranker, validation_set_pairs, qrels, sparse_results, dense_results, feature_version=args.feature_version)
        
        # Print results for each k value
        print("EVALUATION RESULTS:")
        print("=" * 60)
        for k in [10, 100, 1000]:
            print(f"@{k}:")
            print(f"  Baseline  - NDCG: {baseline_results[k]['ndcg']:.4f}, Recall: {baseline_results[k]['recall']:.4f}")
            print(f"  Reranked  - NDCG: {reranked_results[k]['ndcg']:.4f}, Recall: {reranked_results[k]['recall']:.4f}")
            
            # Calculate improvements
            ndcg_improvement = ((reranked_results[k]['ndcg'] - baseline_results[k]['ndcg']) / baseline_results[k]['ndcg'] * 100) if baseline_results[k]['ndcg'] > 0 else 0
            recall_improvement = ((reranked_results[k]['recall'] - baseline_results[k]['recall']) / baseline_results[k]['recall'] * 100) if baseline_results[k]['recall'] > 0 else 0
            
            print(f"  Improvement - NDCG: {ndcg_improvement:+.2f}%, Recall: {recall_improvement:+.2f}%")
            print()
        
        # Save evaluation results to files
        output_dir = f"outputs/models/model_{args.model_version}"
        os.makedirs(output_dir, exist_ok=True)
        save_evaluation_results(baseline_results, reranked_results, args.model_version, args.feature_version, output_dir, args.split)
    else:
        print("Skipping evaluation - missing qrel file or retrieval results")

if __name__ == "__main__":
    main()
