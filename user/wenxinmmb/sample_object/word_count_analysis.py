#!/usr/bin/env python3
"""
Word Count Analysis Script for Wikipedia Articles

This script analyzes the word count distribution of Wikipedia articles from a JSONL corpus file.
It reads each line of the corpus file, extracts the text field, counts words, and provides
statistical analysis with visualization.

Author: wenxinmmb
Date: August 6, 2025
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any
import seaborn as sns
from collections import Counter


def count_words(text: str) -> int:
    """
    Count the number of words in a text string.
    
    Args:
        text (str): The input text
        
    Returns:
        int: Number of words in the text
    """
    if not text or not isinstance(text, str):
        return 0
    
    # Remove extra whitespace and split by whitespace
    words = text.strip().split()
    return len(words)


def load_and_analyze_corpus(file_path: str) -> List[Dict[str, Any]]:
    """
    Load the JSONL corpus file and extract article data with word counts.
    
    Args:
        file_path (str): Path to the JSONL corpus file
        
    Returns:
        List[Dict]: List of dictionaries containing article metadata and word counts
    """
    articles_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    # Parse JSON line
                    article = json.loads(line.strip())
                    
                    # Extract relevant fields
                    article_id = article.get('id', f'unknown_{line_num}')
                    title = article.get('title', 'Unknown Title')
                    text = article.get('text', '')
                    url = article.get('url', '')
                    
                    # Count words in the text
                    word_count = count_words(text)
                    
                    # Store article data
                    articles_data.append({
                        'id': article_id,
                        'title': title,
                        'url': url,
                        'word_count': word_count,
                        'text_length': len(text)
                    })
                    
                    if line_num % 100 == 0:
                        print(f"Processed {line_num} articles...")
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    print(f"Successfully processed {len(articles_data)} articles")
    return articles_data


def calculate_statistics(word_counts: List[int]) -> Dict[str, float]:
    """
    Calculate descriptive statistics for word counts.
    
    Args:
        word_counts (List[int]): List of word counts
        
    Returns:
        Dict[str, float]: Dictionary containing statistical measures
    """
    if not word_counts:
        return {}
    
    word_counts_array = np.array(word_counts)
    
    stats = {
        'count': len(word_counts),
        'mean': np.mean(word_counts_array),
        'median': np.median(word_counts_array),
        'std': np.std(word_counts_array),
        'min': np.min(word_counts_array),
        'max': np.max(word_counts_array),
        'q1': np.percentile(word_counts_array, 25),
        'q3': np.percentile(word_counts_array, 75),
        'iqr': np.percentile(word_counts_array, 75) - np.percentile(word_counts_array, 25)
    }
    
    # Additional percentiles
    for p in [5, 10, 90, 95, 99]:
        stats[f'p{p}'] = np.percentile(word_counts_array, p)
    
    return stats


def plot_word_count_distribution(word_counts: List[int], output_dir: str = "."):
    """
    Create comprehensive visualizations of word count distribution.
    
    Args:
        word_counts (List[int]): List of word counts
        output_dir (str): Directory to save plots
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Wikipedia Articles Word Count Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram with KDE
    axes[0, 0].hist(word_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    axes[0, 0].set_xlabel('Word Count')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Word Count Distribution (Histogram)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add KDE overlay
    from scipy import stats
    density = stats.gaussian_kde(word_counts)
    xs = np.linspace(min(word_counts), max(word_counts), 200)
    axes[0, 0].plot(xs, density(xs), color='red', linewidth=2, label='KDE')
    axes[0, 0].legend()
    
    # 2. Box plot
    axes[0, 1].boxplot(word_counts, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[0, 1].set_ylabel('Word Count')
    axes[0, 1].set_title('Word Count Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Log-scale histogram (for better visualization if there are outliers)
    axes[1, 0].hist(word_counts, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Word Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Word Count Distribution (Log Scale)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_counts = np.sort(word_counts)
    y_vals = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes[1, 1].plot(sorted_counts, y_vals, color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Word Count')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Function')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir) / "word_count_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plots saved to: {output_path}")
    plt.close()  # Close the figure to free memory


def print_summary_statistics(stats: Dict[str, float], articles_data: List[Dict]):
    """
    Print comprehensive summary statistics.
    
    Args:
        stats (Dict[str, float]): Statistical measures
        articles_data (List[Dict]): Article data
    """
    print("\n" + "="*60)
    print("WORD COUNT ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"  Total articles analyzed: {stats['count']:,}")
    
    print(f"\nDescriptive Statistics:")
    print(f"  Mean word count: {stats['mean']:.2f}")
    print(f"  Median word count: {stats['median']:.2f}")
    print(f"  Standard deviation: {stats['std']:.2f}")
    print(f"  Minimum word count: {stats['min']:,}")
    print(f"  Maximum word count: {stats['max']:,}")
    
    print(f"\nQuartiles:")
    print(f"  Q1 (25th percentile): {stats['q1']:.2f}")
    print(f"  Q3 (75th percentile): {stats['q3']:.2f}")
    print(f"  Interquartile Range (IQR): {stats['iqr']:.2f}")
    
    print(f"\nAdditional Percentiles:")
    print(f"  5th percentile: {stats['p5']:.2f}")
    print(f"  10th percentile: {stats['p10']:.2f}")
    print(f"  90th percentile: {stats['p90']:.2f}")
    print(f"  95th percentile: {stats['p95']:.2f}")
    print(f"  99th percentile: {stats['p99']:.2f}")
    
    # Find articles with extreme word counts
    word_counts = [article['word_count'] for article in articles_data]
    
    # Shortest articles
    shortest_articles = sorted(articles_data, key=lambda x: x['word_count'])[:5]
    print(f"\nShortest Articles:")
    for i, article in enumerate(shortest_articles, 1):
        print(f"  {i}. '{article['title']}' ({article['word_count']} words)")
    
    # Longest articles
    longest_articles = sorted(articles_data, key=lambda x: x['word_count'], reverse=True)[:5]
    print(f"\nLongest Articles:")
    for i, article in enumerate(longest_articles, 1):
        print(f"  {i}. '{article['title']}' ({article['word_count']} words)")
    
    # Word count distribution ranges
    print(f"\nWord Count Distribution:")
    ranges = [
        (0, 500, "Very Short"),
        (500, 1000, "Short"),
        (1000, 2500, "Medium"),
        (2500, 5000, "Long"),
        (5000, float('inf'), "Very Long")
    ]
    
    for min_words, max_words, category in ranges:
        if max_words == float('inf'):
            count = sum(1 for wc in word_counts if wc >= min_words)
            print(f"  {category} (≥{min_words} words): {count} articles ({count/len(word_counts)*100:.1f}%)")
        else:
            count = sum(1 for wc in word_counts if min_words <= wc < max_words)
            print(f"  {category} ({min_words}-{max_words-1} words): {count} articles ({count/len(word_counts)*100:.1f}%)")


def save_detailed_analysis(articles_data: List[Dict], output_dir: str = "."):
    """
    Save detailed analysis to CSV files.
    
    Args:
        articles_data (List[Dict]): Article data
        output_dir (str): Directory to save files
    """
    # Create DataFrame
    df = pd.DataFrame(articles_data)
    
    # Save full dataset
    output_path = Path(output_dir) / "word_count_analysis.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Detailed analysis saved to: {output_path}")
    
    # Save summary statistics
    word_counts = df['word_count'].tolist()
    stats = calculate_statistics(word_counts)
    
    stats_df = pd.DataFrame([stats])
    stats_output_path = Path(output_dir) / "word_count_statistics.csv"
    stats_df.to_csv(stats_output_path, index=False)
    print(f"Summary statistics saved to: {stats_output_path}")


def main():
    """
    Main function to run the word count analysis.
    """
    # File path (adjust as needed)
    corpus_file_path = "/home/wenxin/project/data/2025/corpus.jsonl"
    
    # Output directory
    output_dir = "outputs"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Starting Wikipedia Articles Word Count Analysis")
    print(f"Corpus file: {corpus_file_path}")
    print(f"Output directory: {output_dir}")
    
    # Load and analyze the corpus
    articles_data = load_and_analyze_corpus(corpus_file_path)
    
    if not articles_data:
        print("No articles data loaded. Exiting.")
        return
    
    # Extract word counts
    word_counts = [article['word_count'] for article in articles_data]
    
    # Calculate statistics
    stats = calculate_statistics(word_counts)
    
    # Print summary
    print_summary_statistics(stats, articles_data)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_word_count_distribution(word_counts, output_dir)
    
    # Save detailed analysis
    print("\nSaving detailed analysis...")
    save_detailed_analysis(articles_data, output_dir)
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
