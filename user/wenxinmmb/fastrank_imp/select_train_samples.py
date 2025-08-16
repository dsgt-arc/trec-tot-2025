import math
import random
import json
from collections import defaultdict

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

def load_retrieval_results(file_path):
    """Load retrieval results and return dict with query -> doc -> (rank, score)."""
    results = defaultdict(dict)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_id = parts[0]
            doc_id = parts[2]
            rank = int(parts[3])
            score = float(parts[4])
            results[query_id][doc_id] = (rank, score)
    return results

def calculate_reciprocal_rank(rank):
    """Calculate reciprocal rank: 1 / (rank + 1)."""
    return 1.0 / (rank + 1)

def extract_features(doc_id, query_id, sparse_results, dense_results):
    """Extract features for a document given query and retrieval results."""
    features = []
    
    # Get sparse and dense info (default to low values if not found)
    sparse_rank, sparse_score = sparse_results[query_id].get(doc_id, (999, 0.0))
    dense_rank, dense_score = dense_results[query_id].get(doc_id, (999, 0.0))
    
    # Feature 1: sparse retrieval score
    features.append(sparse_score)
    
    # Feature 2: dense retrieval score
    features.append(dense_score)
    
    # Feature 3: math.log1p(bm25_score)
    features.append(math.log1p(sparse_score))
    
    # Feature 4: math.log1p(dense_score + 1)
    features.append(math.log1p(dense_score + 1))
    
    # Feature 5: reciprocal rank for sparse results
    features.append(calculate_reciprocal_rank(sparse_rank))
    
    # Feature 6: reciprocal rank for dense results
    features.append(calculate_reciprocal_rank(dense_rank))
    
    # Feature 7: best rank min(sparse_rank, dense_rank)
    features.append(min(sparse_rank, dense_rank))
    
    # Features 8-9: page view and page rank (to be added later)
    # features.append(0.0)  # placeholder for page view
    # features.append(0.0)  # placeholder for page rank
    
    return features

def categorize_negatives(doc_id, query_id, sparse_results, dense_results, qrels):
    """
    Categorize negative samples as hard or easy.
    
    Hard negatives: Documents that appear in top ranks of either retrieval method
    but are not relevant. These are challenging because the retrieval system 
    thinks they're good matches.
    
    Easy negatives: Documents that appear in lower ranks of both retrieval methods.
    These are easier to distinguish from relevant documents.
    """
    sparse_rank = sparse_results[query_id].get(doc_id, (999, 0.0))[0]
    dense_rank = dense_results[query_id].get(doc_id, (999, 0.0))[0]
    
    # Hard negative: top 100 in either method
    if sparse_rank < 100 or dense_rank < 100:
        return "hard"
    else:
        return "easy"

def sample_training_data(qrels, sparse_results, dense_results, output_file):
    """Sample training data with 1:100 positive to negative ratio."""
    
    training_samples = []
    
    for query_id in qrels:
        relevant_docs = qrels[query_id]
        
        # Get all documents for this query from both retrieval methods
        all_docs = set(sparse_results[query_id].keys()) | set(dense_results[query_id].keys())
        negative_docs = all_docs - relevant_docs
        
        # For each relevant document, sample 100 negatives
        for pos_doc in relevant_docs:
            # Categorize negative documents
            hard_negatives = []
            easy_negatives = []
            
            for neg_doc in negative_docs:
                if categorize_negatives(neg_doc, query_id, sparse_results, dense_results, qrels) == "hard":
                    hard_negatives.append(neg_doc)
                else:
                    easy_negatives.append(neg_doc)
            
            # Sample 70 hard negatives and 30 easy negatives
            sampled_hard = random.sample(hard_negatives, min(70, len(hard_negatives)))
            sampled_easy = random.sample(easy_negatives, min(30, len(easy_negatives)))
            
            # If we don't have enough hard negatives, fill with easy negatives
            if len(sampled_hard) < 70:
                additional_easy = min(100 - len(sampled_hard), len(easy_negatives) - len(sampled_easy))
                sampled_easy.extend(random.sample([doc for doc in easy_negatives if doc not in sampled_easy], additional_easy))
            
            # If we still don't have 100 negatives, fill with remaining negatives
            all_sampled_negatives = sampled_hard + sampled_easy
            if len(all_sampled_negatives) < 100:
                remaining_negatives = [doc for doc in negative_docs if doc not in all_sampled_negatives]
                additional_negatives = min(100 - len(all_sampled_negatives), len(remaining_negatives))
                all_sampled_negatives.extend(random.sample(remaining_negatives, additional_negatives))
            
            # Create positive sample
            pos_features = extract_features(pos_doc, query_id, sparse_results, dense_results)
            training_samples.append((query_id, pos_doc, 1, pos_features))  # label = 1 for relevant
            
            # Create negative samples
            for neg_doc in all_sampled_negatives[:100]:  # Ensure we don't exceed 100
                neg_features = extract_features(neg_doc, query_id, sparse_results, dense_results)
                training_samples.append((query_id, neg_doc, 0, neg_features))  # label = 0 for non-relevant
    
    # Write training samples to file in RankLib format
    with open(output_file, 'w') as f:
        for query_id, doc_id, label, features in training_samples:
            # RankLib format: label qid:query_id 1:feat1 2:feat2 ... # doc_id
            feature_str = " ".join(f"{i+1}:{feat:.6f}" for i, feat in enumerate(features))
            f.write(f"{label} qid:{query_id} {feature_str} # {doc_id}\n")
    
    # Write feature annotations to JSON file
    feature_annotations = {
        "features": {
            "1": {
                "name": "sparse_score",
                "description": "BM25 sparse retrieval score"
            },
            "2": {
                "name": "dense_score", 
                "description": "Dense retrieval score from BGE-M3"
            },
            "3": {
                "name": "log_sparse_score",
                "description": "log1p(BM25 score)"
            },
            "4": {
                "name": "log_dense_score",
                "description": "log1p(dense score + 1)"
            },
            "5": {
                "name": "rr_sparse",
                "description": "Reciprocal rank for sparse retrieval (1/(rank+1))"
            },
            "6": {
                "name": "rr_dense",
                "description": "Reciprocal rank for dense retrieval (1/(rank+1))"
            },
            "7": {
                "name": "best_rank",
                "description": "Best rank between sparse and dense retrieval min(sparse_rank, dense_rank)"
            }
        },
        "format": "RankLib",
        "description": "Features for coordinate ascent reranker training"
    }
    
    feature_file = output_file.replace('.txt', '_features.json')
    with open(feature_file, 'w') as f:
        json.dump(feature_annotations, f, indent=2)
    
    return training_samples

if __name__ == "__main__":
    # File paths
    qrel_file = "/home/wenxin/project/data/2025/generated-queries/llm-set1/dev/qrel.txt"
    sparse_file = "inputs/llm-set1-bm25-run.txt"
    dense_file = "inputs/llm-set1-bge-dense-run.txt"
    output_file = "outputs/training_dev_samples.txt"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load data
    print("Loading qrels...")
    qrels = load_qrels(qrel_file)
    
    print("Loading sparse retrieval results...")
    sparse_results = load_retrieval_results(sparse_file)
    
    print("Loading dense retrieval results...")
    dense_results = load_retrieval_results(dense_file)
    
    # Sample training data
    print("Sampling training data...")
    training_samples = sample_training_data(qrels, sparse_results, dense_results, output_file)
    
    # Print statistics
    total_samples = len(training_samples)
    positive_samples = sum(1 for _, _, label, _ in training_samples if label == 1)
    negative_samples = total_samples - positive_samples
    
    print(f"\nTraining data statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples: {positive_samples}")
    print(f"Negative samples: {negative_samples}")
    print(f"Positive:Negative ratio: 1:{negative_samples//positive_samples}")
    print(f"Training data saved to: {output_file}")
    print(f"Feature annotations saved to: {output_file.replace('.txt', '_features.json')}")
