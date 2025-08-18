import math
import sys

def load_qrels(qrel_file):
    """Load qrels file and return dict of query_id -> dict of doc_id -> relevance."""
    qrels = {}
    with open(qrel_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_id = parts[0]
            doc_id = parts[2]
            relevance = int(parts[3])
            
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = relevance
    return qrels

def load_run_file(run_file):
    """Load run file and return dict of query_id -> list of doc_ids in rank order."""
    runs = {}
    with open(run_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_id = parts[0]
            doc_id = parts[2]
            rank = int(parts[3])
            
            if query_id not in runs:
                runs[query_id] = []
            runs[query_id].append((rank, doc_id))
    
    # Sort by rank for each query
    for query_id in runs:
        runs[query_id].sort(key=lambda x: x[0])
        runs[query_id] = [doc_id for rank, doc_id in runs[query_id]]
    
    return runs

def calculate_recall_at_k(qrels, runs, k):
    """Calculate Recall@K for each query and overall average."""
    recall_scores = {}
    
    for query_id in qrels:
        if query_id not in runs:
            recall_scores[query_id] = 0.0
            continue
        
        # Get relevant documents (relevance > 0)
        relevant_docs = set(doc_id for doc_id, rel in qrels[query_id].items() if rel > 0)
        retrieved_docs = set(runs[query_id][:k])
        
        # Calculate recall: number of relevant docs retrieved / total relevant docs
        relevant_retrieved = len(relevant_docs & retrieved_docs)
        total_relevant = len(relevant_docs)
        
        recall_scores[query_id] = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
    
    # Calculate average recall
    avg_recall = sum(recall_scores.values()) / len(recall_scores) if recall_scores else 0.0
    
    return recall_scores, avg_recall

def calculate_dcg(relevance_scores):
    """Calculate DCG given a list of relevance scores."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because rank starts from 1
    return dcg

def calculate_ndcg_at_k(qrels, runs, k):
    """Calculate NDCG@K for each query and overall average."""
    ndcg_scores = {}
    
    for query_id in qrels:
        if query_id not in runs:
            ndcg_scores[query_id] = 0.0
            continue
        
        # Get retrieved documents up to rank k
        retrieved_docs = runs[query_id][:k]
        
        # Get relevance scores for retrieved documents
        retrieved_relevances = []
        for doc_id in retrieved_docs:
            rel = qrels[query_id].get(doc_id, 0)  # 0 if not in qrels
            retrieved_relevances.append(rel)
        
        # Calculate DCG@k
        dcg_k = calculate_dcg(retrieved_relevances)
        
        # Calculate IDCG@k (ideal DCG)
        all_relevances = list(qrels[query_id].values())
        all_relevances.sort(reverse=True)  # Sort in descending order
        ideal_relevances = all_relevances[:k]  # Take top k
        idcg_k = calculate_dcg(ideal_relevances)
        
        # Calculate NDCG@k
        ndcg_scores[query_id] = dcg_k / idcg_k if idcg_k > 0 else 0.0
    
    # Calculate average NDCG
    avg_ndcg = sum(ndcg_scores.values()) / len(ndcg_scores) if ndcg_scores else 0.0
    
    return ndcg_scores, avg_ndcg

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python manual_scores.py <qrel_file> <run_file>")
        print("Example: python manual_scores.py qrel.txt run.txt")
        sys.exit(1)
    
    qrel_file = sys.argv[1]
    run_file = sys.argv[2]
    k_values = [10, 100, 1000, 2000]
    
    # Load data
    qrels = load_qrels(qrel_file)
    runs = load_run_file(run_file)
    
    print("Evaluation Results:")
    print("=" * 50)
    print(f"Number of queries: {len(qrels)}")
    print()
    
    # Calculate metrics for each k value
    for k in k_values:
        print(f"Metrics @ {k}:")
        print("-" * 30)
        
        # Calculate Recall@K
        recall_scores, avg_recall = calculate_recall_at_k(qrels, runs, k)
        print(f"Recall@{k}: {avg_recall:.4f}")
        
        # Calculate NDCG@K
        ndcg_scores, avg_ndcg = calculate_ndcg_at_k(qrels, runs, k)
        print(f"NDCG@{k}: {avg_ndcg:.4f}")
        print()
    
    # Print detailed results for queries not found in run file
    missing_queries = [q for q in qrels.keys() if q not in runs]
    if missing_queries:
        print(f"Warning: {len(missing_queries)} queries from qrels not found in run file:")
        for q in missing_queries[:10]:  # Show first 10
            print(f"  {q}")
        if len(missing_queries) > 10:
            print(f"  ... and {len(missing_queries) - 10} more")
