"""
Comprehensive Retrieval Results Visualization Script
Creates visualizations comparing LLM, Sparse (BM25), and Dense (BGE-M3) retrieval results
for original queries vs. different LLM-generated queries.
Also includes comparison of 5 dense retrieval datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import pi

def create_all_retrieval_data():
    """
    Create DataFrames with all retrieval results from trec_eval outputs.
    Returns LLM, BM25, and Dense retrieval results.
    """
    
    # LLM retrieval results (Gemini-2.5-Flash)
    llm_data = {
        'System': [
            'Original Queries',
            'Gemini-2.5-Flash-Lite',
            'GPT-4o-Mini', 
            'GPT-4o-2024-08-06'
        ],
        'Reciprocal Rank': [0.4381, 0.3797, 0.3860, 0.2895],
        'Recall@1000': [0.6400, 0.6000, 0.6100, 0.4800],
        'NDCG@10': [0.4792, 0.4211, 0.4283, 0.3185],
        'NDCG@1000': [0.4862, 0.4311, 0.4382, 0.3340]
    }
    
    # PyTerrier BM25 retrieval results (Sparse)
    sparse_data = {
        'System': [
            'Original Queries',
            'Gemini-2.5-Flash-Lite',
            'GPT-4o-Mini', 
            'GPT-4o-2024-08-06'
        ],
        'Reciprocal Rank': [0.3038, 0.0896, 0.0913, 0.0754],
        'Recall@1000': [0.8100, 0.5500, 0.6100, 0.5900],
        'NDCG@10': [0.3302, 0.0999, 0.0899, 0.0826],
        'NDCG@1000': [0.3894, 0.1583, 0.1657, 0.1524]
    }
    
    # BGE-M3 Dense retrieval results (first 4 systems)
    dense_data = {
        'System': [
            'Original Queries',
            'Gemini-2.5-Flash-Lite',
            'GPT-4o-Mini', 
            'GPT-4o-2024-08-06'
        ],
        'Reciprocal Rank': [0.1545, 0.0590, 0.0644, 0.0366],
        'Recall@1000': [0.6400, 0.5500, 0.4800, 0.3300],
        'NDCG@10': [0.1667, 0.0580, 0.0669, 0.0375],
        'NDCG@1000': [0.2315, 0.1337, 0.1224, 0.0822]
    }
    
    # Complete Dense retrieval results (all 5 datasets)
    dense_complete_data = {
        'System': [
            'Original Queries',
            'Gemini-2.5-Flash-Lite',
            'GPT-4o-Mini', 
            'GPT-4o-2024-08-06',
            'Wikipedia Text (first 200 words)',
            'Random Text'
        ],
        'Reciprocal Rank': [0.1545, 0.0590, 0.0644, 0.0366, 0.9950, 0.0000],
        'Recall@1000': [0.6400, 0.5500, 0.4800, 0.3300, 1.0000, 0.0100],
        'NDCG@10': [0.1667, 0.0580, 0.0669, 0.0375, 0.9963, 0.0000],
        'NDCG@1000': [0.2315, 0.1337, 0.1224, 0.0822, 0.9963, 0.0010]
    }
    
    llm_df = pd.DataFrame(llm_data)
    sparse_df = pd.DataFrame(sparse_data)
    dense_df = pd.DataFrame(dense_data)
    dense_complete_df = pd.DataFrame(dense_complete_data)
    
    return llm_df, sparse_df, dense_df, dense_complete_df

def plot_three_method_comparison(llm_df, sparse_df, dense_df):
    """
    Create comprehensive comparison of all three retrieval methods.
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Retrieval Performance Comparison: LLM vs Sparse (BM25) vs Dense (BGE-M3)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    metrics = ['Reciprocal Rank', 'Recall@1000', 'NDCG@10', 'NDCG@1000']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Top row: Individual method performance
    for i, metric in enumerate(metrics):
        x = np.arange(len(llm_df['System']))
        width = 0.25
        
        bars1 = axes[0, i].bar(x - width, llm_df[metric], width, 
                              label='LLM Retrieval', color=colors[i], alpha=0.9)
        bars2 = axes[0, i].bar(x, sparse_df[metric], width,
                              label='Sparse (BM25)', color=colors[i], alpha=0.6)
        bars3 = axes[0, i].bar(x + width, dense_df[metric], width,
                              label='Dense (BGE-M3)', color=colors[i], alpha=0.3)
        
        axes[0, i].set_title(f'{metric}', fontsize=12, fontweight='bold')
        axes[0, i].set_ylabel('Score', fontsize=10)
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels(llm_df['System'], rotation=45, ha='right')
        axes[0, i].legend()
        axes[0, i].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.001:  # Only label if value is significant
                    axes[0, i].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Bottom row: Performance degradation from original
    for i, metric in enumerate(metrics):
        original_scores = {
            'LLM': llm_df.iloc[0][metric],
            'Sparse': sparse_df.iloc[0][metric],
            'Dense': dense_df.iloc[0][metric]
        }
        
        systems = llm_df['System'][1:].tolist()  # Skip original
        x = np.arange(len(systems))
        width = 0.25
        
        # Calculate relative performance (as percentage of original)
        llm_relative = [(llm_df.iloc[j+1][metric] / original_scores['LLM']) * 100 for j in range(len(systems))]
        sparse_relative = [(sparse_df.iloc[j+1][metric] / original_scores['Sparse']) * 100 for j in range(len(systems))]
        dense_relative = [(dense_df.iloc[j+1][metric] / original_scores['Dense']) * 100 for j in range(len(systems))]
        
        bars1 = axes[1, i].bar(x - width, llm_relative, width, 
                              label='LLM Retrieval', color=colors[i], alpha=0.9)
        bars2 = axes[1, i].bar(x, sparse_relative, width,
                              label='Sparse (BM25)', color=colors[i], alpha=0.6)
        bars3 = axes[1, i].bar(x + width, dense_relative, width,
                              label='Dense (BGE-M3)', color=colors[i], alpha=0.3)
        
        axes[1, i].set_title(f'{metric} (% of Original)', fontsize=12, fontweight='bold')
        axes[1, i].set_ylabel('Percentage of Original Score', fontsize=10)
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels(systems, rotation=45, ha='right')
        axes[1, i].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Original Performance')
        axes[1, i].legend()
        axes[1, i].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                axes[1, i].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('comprehensive_retrieval_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_dense_complete_analysis(dense_complete_df):
    """
    Create performance heatmap for all 5 dense retrieval datasets.
    """
    plt.style.use('default')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle('Dense Retrieval (BGE-M3) Performance Heatmap', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    metrics = ['Reciprocal Rank', 'Recall@1000', 'NDCG@10', 'NDCG@1000']
    
    # Create heatmap data
    heatmap_data = dense_complete_df.set_index('System')[metrics]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Score'})
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Systems', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('dense_retrieval_complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_comprehensive_analysis(llm_df, sparse_df, dense_df, dense_complete_df):
    """
    Print detailed analysis of all retrieval results.
    """
    print("=" * 80)
    print("COMPREHENSIVE RETRIEVAL ANALYSIS")
    print("=" * 80)
    
    # Method performance ranking
    print("\n1. OVERALL METHOD PERFORMANCE RANKING:")
    print("-" * 40)
    
    metrics = ['Reciprocal Rank', 'Recall@1000', 'NDCG@10', 'NDCG@1000']
    
    # For original queries
    original_scores = {
        'LLM Retrieval': llm_df.iloc[0][metrics].mean(),
        'Sparse (BM25)': sparse_df.iloc[0][metrics].mean(),
        'Dense (BGE-M3)': dense_df.iloc[0][metrics].mean()
    }
    
    sorted_methods = sorted(original_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (method, score) in enumerate(sorted_methods, 1):
        print(f"{i}. {method}: {score:.4f}")
    
    # Performance degradation analysis
    print("\n2. PERFORMANCE DEGRADATION ANALYSIS:")
    print("-" * 40)
    
    systems = ['Gemini-2.5-Flash-Lite', 'GPT-4o-Mini', 'GPT-4o-2024-08-06']
    
    for system in systems:
        idx = llm_df[llm_df['System'] == system].index[0]
        
        llm_deg = (1 - llm_df.iloc[idx][metrics].mean() / llm_df.iloc[0][metrics].mean()) * 100
        sparse_deg = (1 - sparse_df.iloc[idx][metrics].mean() / sparse_df.iloc[0][metrics].mean()) * 100
        dense_deg = (1 - dense_df.iloc[idx][metrics].mean() / dense_df.iloc[0][metrics].mean()) * 100
        
        print(f"\n{system}:")
        print(f"  LLM Retrieval degradation: {llm_deg:.1f}%")
        print(f"  Sparse (BM25) degradation: {sparse_deg:.1f}%")
        print(f"  Dense (BGE-M3) degradation: {dense_deg:.1f}%")
    
    # Dense retrieval complete analysis
    print("\n3. DENSE RETRIEVAL COMPLETE ANALYSIS:")
    print("-" * 40)
    
    # Find best and worst performers in dense retrieval
    dense_avg_scores = [dense_complete_df.iloc[i][metrics].mean() for i in range(len(dense_complete_df))]
    best_idx = np.argmax(dense_avg_scores)
    worst_idx = np.argmin(dense_avg_scores)
    
    print(f"Best performer: {dense_complete_df.iloc[best_idx]['System']} (avg: {dense_avg_scores[best_idx]:.4f})")
    print(f"Worst performer: {dense_complete_df.iloc[worst_idx]['System']} (avg: {dense_avg_scores[worst_idx]:.4f})")
    
    # Performance gap analysis
    wiki_score = dense_complete_df.iloc[4][metrics].mean()  # Wikipedia text
    original_score = dense_complete_df.iloc[0][metrics].mean()  # Original queries
    
    print(f"\nWikipedia text vs Original queries performance gap: {((wiki_score / original_score) - 1) * 100:.1f}%")
    
    # Recommendations
    print("\n4. KEY INSIGHTS AND RECOMMENDATIONS:")
    print("-" * 40)
    
    print("• LLM retrieval shows the most robust performance across generated queries")
    print("• Sparse retrieval (BM25) is most sensitive to query reformulation")
    print("• Dense retrieval shows extreme sensitivity to query type (Wikipedia text vs others)")
    print("• GPT-4o-2024-08-06 consistently shows the highest degradation across all methods")
    print("• Gemini-2.5-Flash-Lite and GPT-4o-Mini show similar, more stable performance")

def save_all_results(llm_df, sparse_df, dense_df, dense_complete_df):
    """
    Save all results to CSV files.
    """
    import os
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Save individual method results
    llm_df.to_csv('outputs/llm_retrieval_results.csv', index=False)
    sparse_df.to_csv('outputs/sparse_retrieval_results.csv', index=False)
    dense_df.to_csv('outputs/dense_retrieval_results.csv', index=False)
    dense_complete_df.to_csv('outputs/dense_complete_retrieval_results.csv', index=False)
    
    # Create combined results for easy comparison
    combined_df = pd.DataFrame({
        'System': llm_df['System'],
        'LLM_RR': llm_df['Reciprocal Rank'],
        'LLM_Recall': llm_df['Recall@1000'],
        'LLM_NDCG10': llm_df['NDCG@10'],
        'LLM_NDCG1000': llm_df['NDCG@1000'],
        'Sparse_RR': sparse_df['Reciprocal Rank'],
        'Sparse_Recall': sparse_df['Recall@1000'],
        'Sparse_NDCG10': sparse_df['NDCG@10'],
        'Sparse_NDCG1000': sparse_df['NDCG@1000'],
        'Dense_RR': dense_df['Reciprocal Rank'],
        'Dense_Recall': dense_df['Recall@1000'],
        'Dense_NDCG10': dense_df['NDCG@10'],
        'Dense_NDCG1000': dense_df['NDCG@1000']
    })
    
    combined_df.to_csv('outputs/combined_retrieval_results.csv', index=False)
    
    print("\nSaved files:")
    print("- outputs/llm_retrieval_results.csv")
    print("- outputs/sparse_retrieval_results.csv") 
    print("- outputs/dense_retrieval_results.csv")
    print("- outputs/dense_complete_retrieval_results.csv")
    print("- outputs/combined_retrieval_results.csv")

if __name__ == "__main__":
    print("Creating comprehensive retrieval results visualizations...")
    print("=" * 60)
    
    # Load all data
    llm_df, sparse_df, dense_df, dense_complete_df = create_all_retrieval_data()
    
    # Create visualizations
    print("\n1. Creating three-method comparison visualization...")
    fig1 = plot_three_method_comparison(llm_df, sparse_df, dense_df)
    
    print("\n2. Creating dense retrieval complete analysis...")
    fig2 = plot_dense_complete_analysis(dense_complete_df)
    
    # Print analysis
    print_comprehensive_analysis(llm_df, sparse_df, dense_df, dense_complete_df)
    
    # Save results
    save_all_results(llm_df, sparse_df, dense_df, dense_complete_df)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION FILES CREATED:")
    print("=" * 60)
    print("- comprehensive_retrieval_comparison.png")
    print("- dense_retrieval_complete_analysis.png") 
    print("- method_comparison_summary.png")
    print("\nAll visualizations complete!")
