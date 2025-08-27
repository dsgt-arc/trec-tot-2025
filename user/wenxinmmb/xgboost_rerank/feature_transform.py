import os
import sqlite3
import pandas as pd
import numpy as np  # Import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import json  # Import json module
import glob

# Global variables to cache data loaders
_pageview_cache = {}
_pagerank_cache = {}

def load_pageview_data():
    """Load pageview data from SQLite database, with caching."""
    global _pageview_cache
    if _pageview_cache:
        return _pageview_cache
    
    # Get TOT path from environment or use default
    tot_path = os.getenv('TOT', '/home/wenxin/project-v2/trec-tot-2025')
    db_path = os.path.join(tot_path, 'sample_object/outputs/wikipedia_data.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT wikipedia_id, page_views FROM wikipedia_articles")
    
    for wikipedia_id, page_views in cursor.fetchall():
        _pageview_cache[str(wikipedia_id)] = page_views if page_views is not None else 0
    
    conn.close()
    print(f"Loaded {len(_pageview_cache)} pageview records from {db_path}")

    return _pageview_cache

def load_pagerank_data():
    """Load pagerank data from parquet file, with caching."""
    global _pagerank_cache
    if _pagerank_cache:
        return _pagerank_cache
    
    pagerank_file = os.path.join('/home/wenxin/project', 'merged-graph/pagerank.parquet')
    df = pd.read_parquet(pagerank_file)
    _pagerank_cache = dict(zip(df['id'].astype(str), df['pagerank'].fillna(0.0)))
    print(f"Loaded {len(_pagerank_cache)} pagerank records from {pagerank_file}")

    return _pagerank_cache

def plot_data_distribution():
    """Plot the distribution of pageview and pagerank data after normalization and print statistics."""
    # Load data
    pageview_data = load_pageview_data()
    pagerank_data = load_pagerank_data()

    # Convert to lists for processing
    pageview_values = list(pageview_data.values())
    pagerank_values = list(pagerank_data.values())

    # Log normalization (log1p for pageviews and pagerank)
    pageview_log = pd.Series(pageview_values).apply(lambda x: np.log1p(x))
    pagerank_log = pd.Series(pagerank_values).apply(lambda x: np.log(x))

    # Z-score normalization
    pageview_zscore = (pageview_log - pageview_log.mean()) / pageview_log.std()
    pagerank_zscore = (pagerank_log - pagerank_log.mean()) / pagerank_log.std()

    # Save normalized scores to JSON files
    normalized_pageview = {k: v for k, v in zip(pageview_data.keys(), pageview_zscore)}
    normalized_pagerank = {k: v for k, v in zip(pagerank_data.keys(), pagerank_zscore)}

    output_dir = 'outputs/normalized-features'
    os.makedirs(output_dir, exist_ok=True)

    pageview_file = os.path.join(output_dir, 'normalized_pageview.json')
    pagerank_file = os.path.join(output_dir, 'normalized_pagerank.json')

    with open(pageview_file, 'w') as f:
        json.dump(normalized_pageview, f, indent=4)
    print(f"Normalized pageview scores saved to {pageview_file}")

    with open(pagerank_file, 'w') as f:
        json.dump(normalized_pagerank, f, indent=4)
    print(f"Normalized pagerank scores saved to {pagerank_file}")

    # Calculate and print statistics for normalized pageview data
    print("Pageview (Log1p + Z-Score) Statistics:")
    print(f"  Min: {pageview_zscore.min():.2f}")
    print(f"  Max: {pageview_zscore.max():.2f}")
    print(f"  Mean: {pageview_zscore.mean():.2f}")
    print(f"  Std: {pageview_zscore.std():.2f}")
    print(f"  Count: {pageview_zscore.count():.0f}")

    # Calculate and print statistics for normalized pagerank data
    print("Pagerank (Log + Z-Score) Statistics:")
    print(f"  Min: {pagerank_zscore.min():.6f}")
    print(f"  Max: {pagerank_zscore.max():.6f}")
    print(f"  Mean: {pagerank_zscore.mean():.6f}")
    print(f"  Std: {pagerank_zscore.std():.6f}")
    print(f"  Count: {pagerank_zscore.count():.0f}")

    # Plot distributions
    plt.figure(figsize=(12, 6))

    # Pageview distribution
    plt.subplot(1, 2, 1)
    sns.histplot(pageview_zscore, kde=True, bins=50, color='blue')
    plt.title('Pageview (Log1p + Z-Score) Distribution')
    plt.xlabel('Z-Score')
    plt.ylabel('Frequency')

    # Pagerank distribution
    plt.subplot(1, 2, 2)
    sns.histplot(pagerank_zscore, kde=True, bins=50, color='green')
    plt.title('Pagerank (Log + Z-Score) Distribution')
    plt.xlabel('Z-Score')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def min_max_normalize_pagerank():
    """Perform min-max normalization on pagerank data, save the results to a JSON file, and visualize the distribution."""
    # Load pagerank data
    pagerank_data = load_pagerank_data()
    pagerank_values = np.array(list(pagerank_data.values()))

    # Min-max normalization
    min_val = pagerank_values.min()
    max_val = pagerank_values.max()
    pagerank_normalized = (pagerank_values - min_val) / (max_val - min_val)

    # Create a dictionary with normalized values
    normalized_pagerank = {k: v for k, v in zip(pagerank_data.keys(), pagerank_normalized)}

    # Save to JSON file
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs/normalized-features')
    os.makedirs(output_dir, exist_ok=True)
    pagerank_file = os.path.join(output_dir, 'min_max_normalized_pagerank.json')

    with open(pagerank_file, 'w') as f:
        json.dump(normalized_pagerank, f, indent=2)
    print(f"Min-max normalized pagerank scores saved to {pagerank_file}")

    # Print statistics
    print("Pagerank (Min-Max Normalized) Statistics:")
    print(f"  Min: {pagerank_normalized.min():.6f}")
    print(f"  Max: {pagerank_normalized.max():.6f}")
    print(f"  Mean: {pagerank_normalized.mean():.6f}")
    print(f"  Std: {pagerank_normalized.std():.6f}")
    print(f"  Count: {len(pagerank_normalized)}")

    # Visualize the distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(pagerank_normalized, kde=True, bins=50, color='purple')
    plt.title('Pagerank (Min-Max Normalized) Distribution')
    plt.xlabel('Normalized Score')
    plt.ylabel('Frequency')
    plt.show()

def min_max_and_zscore_normalize_pagerank():
    """Perform min-max normalization followed by z-score normalization on pagerank data, save the results, and visualize the distribution."""
    # Load pagerank data
    pagerank_data = load_pagerank_data()
    pagerank_values = np.array(list(pagerank_data.values()))

    # Min-max normalization
    min_val = pagerank_values.min()
    max_val = pagerank_values.max()
    pagerank_min_max = (pagerank_values - min_val) / (max_val - min_val)

    # Z-score normalization
    mean_val = pagerank_min_max.mean()
    std_val = pagerank_min_max.std()
    pagerank_zscore = (pagerank_min_max - mean_val) / std_val

    # Create dictionaries with normalized values
    normalized_min_max_pagerank = {k: v for k, v in zip(pagerank_data.keys(), pagerank_min_max)}
    normalized_zscore_pagerank = {k: v for k, v in zip(pagerank_data.keys(), pagerank_zscore)}

    # Save to JSON files
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs/normalized-features')
    os.makedirs(output_dir, exist_ok=True)

    min_max_file = os.path.join(output_dir, 'min_max_normalized_pagerank.json')
    zscore_file = os.path.join(output_dir, 'zscore_normalized_pagerank.json')

    with open(min_max_file, 'w') as f:
        json.dump(normalized_min_max_pagerank, f, indent=2)
    print(f"Min-max normalized pagerank scores saved to {min_max_file}")

    with open(zscore_file, 'w') as f:
        json.dump(normalized_zscore_pagerank, f, indent=2)
    print(f"Z-score normalized pagerank scores saved to {zscore_file}")

    # Print statistics for min-max normalization
    print("Pagerank (Min-Max Normalized) Statistics:")
    print(f"  Min: {pagerank_min_max.min():.6f}")
    print(f"  Max: {pagerank_min_max.max():.6f}")
    print(f"  Mean: {pagerank_min_max.mean():.6f}")
    print(f"  Std: {pagerank_min_max.std():.6f}")
    print(f"  Count: {len(pagerank_min_max)}")

    # Print statistics for z-score normalization
    print("Pagerank (Z-Score Normalized) Statistics:")
    print(f"  Min: {pagerank_zscore.min():.6f}")
    print(f"  Max: {pagerank_zscore.max():.6f}")
    print(f"  Mean: {pagerank_zscore.mean():.6f}")
    print(f"  Std: {pagerank_zscore.std():.6f}")
    print(f"  Count: {len(pagerank_zscore)}")

    # Visualize the distributions
    plt.figure(figsize=(12, 6))

    # Min-max normalized distribution
    plt.subplot(1, 2, 1)
    sns.histplot(pagerank_min_max, kde=True, bins=50, color='blue')
    plt.title('Pagerank (Min-Max Normalized) Distribution')
    plt.xlabel('Normalized Score')
    plt.ylabel('Frequency')

    # Z-score normalized distribution
    plt.subplot(1, 2, 2)
    sns.histplot(pagerank_zscore, kde=True, bins=50, color='green')
    plt.title('Pagerank (Z-Score Normalized) Distribution')
    plt.xlabel('Normalized Score')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def calculate_query_word_count_stats():
    """Calculate word count statistics for queries in JSONL files and perform min-max normalization."""

    query_files = [
        '/home/wenxin/project/data/2025/train-2025/queries.jsonl',
        '/home/wenxin/project/data/2025/dev1-2025/queries.jsonl',
        '/home/wenxin/project/data/2025/dev2-2025/queries.jsonl',
        '/home/wenxin/project/data/2025/dev3-2025/queries.jsonl',
    ]

    print(query_files)
    word_counts = []

    # Process each query file
    for query_file in query_files:
        with open(query_file, 'r') as f:
            for line in f:
                query_data = json.loads(line)
                query = query_data.get('query', '')
                word_count = len(query.split())
                word_counts.append(word_count)

    # Convert to numpy array for calculations
    word_counts = np.array(word_counts)

    # Calculate statistics
    min_count = word_counts.min()
    max_count = word_counts.max()
    mean_count = word_counts.mean()
    std_count = word_counts.std()

    # Print statistics
    print("Query Word Count Statistics:")
    print(f"  Min: {min_count}")
    print(f"  Max: {max_count}")
    print(f"  Mean: {mean_count:.2f}")
    print(f"  Std: {std_count:.2f}")
    print(f"  Total Queries: {len(word_counts)}")

    # Perform min-max normalization
    normalized_word_counts = (word_counts - min_count) / (max_count - min_count)

    # Plot the normalized word counts
    plt.figure(figsize=(8, 6))
    sns.histplot(normalized_word_counts, kde=True, bins=50, color='orange')
    plt.title('Normalized Query Word Count Distribution (Min-Max)')
    plt.xlabel('Normalized Word Count')
    plt.ylabel('Frequency')
    plt.show()
    

# Example usage
if __name__ == "__main__":
    # plot_data_distribution()
    # min_max_normalize_pagerank()
    # min_max_and_zscore_normalize_pagerank()
    calculate_query_word_count_stats()