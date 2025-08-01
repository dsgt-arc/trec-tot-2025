import numpy as np
import json
from scipy.stats import pearsonr, kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# embedding_model = 'all-MiniLM-L12-v2'  # Pre-trained model for sentence embeddings
embedding_model = 'all-MiniLM-L6-v2'  # Pre-trained model for sentence embeddings
def correlation_embeddings(queries1, queries2, method='sentence_transformer'):
    """
    Calculate Pearson and Kendall's Tau correlation between two sets of queries using embeddings.
    
    Args:
        queries1 (list): First set of query strings
        queries2 (list): Second set of query strings  
        method (str): 'sentence_transformer', 'tfidf', or 'word2vec'
    
    Returns:
        dict: Dictionary containing both correlation coefficients and p-values
    """
    
    if method == 'sentence_transformer':
        # Using pre-trained sentence transformer
        model = SentenceTransformer(embedding_model)
        embeddings1 = model.encode(queries1)
        embeddings2 = model.encode(queries2)
        
    elif method == 'tfidf':
        # Using TF-IDF vectors
        vectorizer = TfidfVectorizer()
        all_queries = queries1 + queries2
        tfidf_matrix = vectorizer.fit_transform(all_queries)
        
        embeddings1 = tfidf_matrix[:len(queries1)].toarray()
        embeddings2 = tfidf_matrix[len(queries1):].toarray()
    
    # Calculate mean embeddings for each set
    mean_embedding1 = np.mean(embeddings1, axis=0)
    mean_embedding2 = np.mean(embeddings2, axis=0)
    
    # Calculate Pearson correlation
    pearson_corr, pearson_p = pearsonr(mean_embedding1, mean_embedding2)
    
    # Calculate Kendall's Tau correlation
    kendall_corr, kendall_p = kendalltau(mean_embedding1, mean_embedding2)
    
    return {
        'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
        'kendall': {'correlation': kendall_corr, 'p_value': kendall_p}
    }

def pearson_correlation_embeddings(queries1, queries2, method='sentence_transformer'):
    """
    Legacy function for backward compatibility. Returns only Pearson correlation.
    
    Args:
        queries1 (list): First set of query strings
        queries2 (list): Second set of query strings  
        method (str): 'sentence_transformer', 'tfidf', or 'word2vec'
    
    Returns:
        tuple: (correlation, p_value)
    """
    results = correlation_embeddings(queries1, queries2, method)
    return results['pearson']['correlation'], results['pearson']['p_value']

def load_queries_from_jsonl(file_path):
    """
    Load queries from JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        list: List of query strings
    """
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            queries.append(data['query'])
    return queries

# Load queries from JSONL files
path1 = '/home/wenxin/project/data/2025/dev3-2025/queries-first-100.jsonl'
path2 = 'outputs/random_200_queries.jsonl'

queries_set1 = load_queries_from_jsonl(path1)
# queries_set2 = load_queries_from_jsonl('/home/wenxin/project/data/2025/dev3-g1-2025/queries.jsonl')
queries_set2 = load_queries_from_jsonl(path2)

print(f"Loaded {len(queries_set1)} queries from {path1}")
print(f"Loaded {len(queries_set2)} queries from {path2}")

results = correlation_embeddings(queries_set1, queries_set2)
print(f"Pearson correlation: {results['pearson']['correlation']:.4f}")
print(f"Pearson P-value: {results['pearson']['p_value']:.4f}")
print(f"Kendall's Tau correlation: {results['kendall']['correlation']:.4f}")
print(f"Kendall's Tau P-value: {results['kendall']['p_value']:.4f}")


# queries_set3 = load_queries_from_jsonl('/home/wenxin/project/data/2025/dev3-o4-2025/queries.jsonl')

# print(f"\nLoaded {len(queries_set1)} queries from dev3-2025")
# print(f"Loaded {len(queries_set3)} queries from dev3-o4-2025")

# results = correlation_embeddings(queries_set1, queries_set3)
# print(f"Pearson correlation set 1 and set 3: {results['pearson']['correlation']:.4f}")
# print(f"Pearson P-value: {results['pearson']['p_value']:.4f}")
# print(f"Kendall's Tau correlation set 1 and set 3: {results['kendall']['correlation']:.4f}")
# print(f"Kendall's Tau P-value: {results['kendall']['p_value']:.4f}")

print(f'Embedding used: {embedding_model}')