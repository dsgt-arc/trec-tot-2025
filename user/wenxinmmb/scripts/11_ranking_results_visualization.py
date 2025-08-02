import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_ranking_results_data():
    """
    Create DataFrames with the ranking results from trec_eval outputs.
    Returns both LLM retrieval and PyTerrier BM25 results.
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
    
    # PyTerrier BM25 retrieval results
    bm25_data = {
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
    
    llm_df = pd.DataFrame(llm_data)
    bm25_df = pd.DataFrame(bm25_data)
    
    return llm_df, bm25_df

def plot_ranking_metrics(llm_df, bm25_df):
    """
    Create comprehensive visualizations of the ranking metrics for both retrieval methods.
    """
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots for both retrieval methods
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('TREC Evaluation Results: Query Performance Comparison\nLLM Retrieval vs PyTerrier BM25', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Metrics to plot
    metrics = ['Reciprocal Rank', 'Recall@1000', 'NDCG@10', 'NDCG@1000']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Plot LLM retrieval results (top row)
    for i, metric in enumerate(metrics):
        bars = axes[0, i].bar(llm_df['System'], llm_df[metric], color=colors[i], alpha=0.8)
        axes[0, i].set_title(f'{metric}\n(LLM Retrieval)', fontsize=11, fontweight='bold')
        axes[0, i].set_ylabel('Score', fontsize=10)
        axes[0, i].tick_params(axis='x', rotation=45)
        axes[0, i].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, i].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot BM25 retrieval results (middle row)
    for i, metric in enumerate(metrics):
        bars = axes[1, i].bar(bm25_df['System'], bm25_df[metric], color=colors[i], alpha=0.8)
        axes[1, i].set_title(f'{metric}\n(PyTerrier BM25)', fontsize=11, fontweight='bold')
        axes[1, i].set_ylabel('Score', fontsize=10)
        axes[1, i].tick_params(axis='x', rotation=45)
        axes[1, i].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, i].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Comparison plots (bottom row)
    x = np.arange(len(llm_df['System']))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        # Side-by-side comparison
        bars1 = axes[2, i].bar(x - width/2, llm_df[metric], width, 
                              label='LLM Retrieval', color=colors[i], alpha=0.8)
        bars2 = axes[2, i].bar(x + width/2, bm25_df[metric], width,
                              label='PyTerrier BM25', color=colors[i], alpha=0.5)
        
        axes[2, i].set_title(f'{metric}\nComparison', fontsize=11, fontweight='bold')
        axes[2, i].set_ylabel('Score', fontsize=10)
        axes[2, i].set_xticks(x)
        axes[2, i].set_xticklabels(llm_df['System'], rotation=45, ha='right')
        axes[2, i].legend()
        axes[2, i].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[2, i].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            axes[2, i].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('ranking_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_performance_radar(llm_df, bm25_df):
    """
    Create radar charts to compare system performance across all metrics for both retrieval methods.
    """
    
    # Prepare data for radar chart
    metrics = ['Reciprocal Rank', 'Recall@1000', 'NDCG@10', 'NDCG@1000']
    systems = llm_df['System'].tolist()
    
    # Number of metrics
    N = len(metrics)
    
    # Compute angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the radar plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # LLM Retrieval radar chart
    for i, system in enumerate(systems):
        values = llm_df.iloc[i, 1:].values.tolist()  # Get metric values
        values += values[:1]  # Complete the circle
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=system, color=colors[i])
        ax1.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Add metric labels
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.set_ylim(0, 0.7)
    ax1.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax1.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], fontsize=9)
    ax1.grid(True)
    ax1.set_title('LLM Retrieval Performance', size=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # PyTerrier BM25 radar chart
    for i, system in enumerate(systems):
        values = bm25_df.iloc[i, 1:].values.tolist()  # Get metric values
        values += values[:1]  # Complete the circle
        
        ax2.plot(angles, values, 'o-', linewidth=2, label=system, color=colors[i])
        ax2.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Add metric labels
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.set_ylim(0, 0.9)
    ax2.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax2.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'], fontsize=9)
    ax2.grid(True)
    ax2.set_title('PyTerrier BM25 Performance', size=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('ranking_results_radar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_relative_performance(llm_df, bm25_df):
    """
    Create plots showing relative performance compared to the original queries for both retrieval methods.
    """
    
    # Calculate relative performance for both methods
    def calculate_relative_performance(df):
        original_idx = 0  # Original queries are at index 0
        metrics = ['Reciprocal Rank', 'Recall@1000', 'NDCG@10', 'NDCG@1000']
        
        relative_data = []
        for i, system in enumerate(df['System']):
            if i == original_idx:
                # Original is 100% for all metrics
                relative_row = [system] + [100.0] * len(metrics)
            else:
                relative_row = [system]
                for metric in metrics:
                    original_value = df.iloc[original_idx][metric]
                    current_value = df.iloc[i][metric]
                    relative_performance = (current_value / original_value) * 100
                    relative_row.append(relative_performance)
            relative_data.append(relative_row)
        
        return pd.DataFrame(relative_data, columns=['System'] + metrics)
    
    llm_relative_df = calculate_relative_performance(llm_df)
    bm25_relative_df = calculate_relative_performance(bm25_df)
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    metrics = ['Reciprocal Rank', 'Recall@1000', 'NDCG@10', 'NDCG@1000']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # LLM Retrieval relative performance
    x = np.arange(len(llm_relative_df['System']))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, llm_relative_df[metric], width, label=metric, 
                      color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if j > 0:  # Don't show 100% for original
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, 
                label='Original Performance (100%)')
    ax1.set_title('LLM Retrieval: Relative Performance vs Original', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance (% of Original)', fontsize=10)
    ax1.set_xlabel('System', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(llm_relative_df['System'], rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    # PyTerrier BM25 relative performance
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax2.bar(x + offset, bm25_relative_df[metric], width, label=metric, 
                      color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if j > 0:  # Don't show 100% for original
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, 
                label='Original Performance (100%)')
    ax2.set_title('PyTerrier BM25: Relative Performance vs Original', 
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance (% of Original)', fontsize=10)
    ax2.set_xlabel('System', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bm25_relative_df['System'], rotation=45, ha='right')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ranking_results_relative.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, llm_relative_df, bm25_relative_df

def print_ranking_summary(llm_df, bm25_df):
    """
    Print a detailed summary of the ranking results for both retrieval methods.
    """
    print("=" * 80)
    print("TREC EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    print("LLM RETRIEVAL PERFORMANCE (Gemini-2.5-Flash):")
    print("-" * 50)
    for _, row in llm_df.iterrows():
        print(f"{row['System']}:")
        print(f"  Reciprocal Rank: {row['Reciprocal Rank']:.4f}")
        print(f"  Recall@1000:     {row['Recall@1000']:.4f}")
        print(f"  NDCG@10:         {row['NDCG@10']:.4f}")
        print(f"  NDCG@1000:       {row['NDCG@1000']:.4f}")
        print()
    
    print("PYTERRIER BM25 RETRIEVAL PERFORMANCE:")
    print("-" * 50)
    for _, row in bm25_df.iterrows():
        print(f"{row['System']}:")
        print(f"  Reciprocal Rank: {row['Reciprocal Rank']:.4f}")
        print(f"  Recall@1000:     {row['Recall@1000']:.4f}")
        print(f"  NDCG@10:         {row['NDCG@10']:.4f}")
        print(f"  NDCG@1000:       {row['NDCG@1000']:.4f}")
        print()
    
    print("COMPARATIVE ANALYSIS:")
    print("-" * 50)
    
    # Compare LLM vs BM25 for original queries
    print("Original Queries Performance Comparison:")
    llm_orig = llm_df.iloc[0]
    bm25_orig = bm25_df.iloc[0]
    metrics = ['Reciprocal Rank', 'Recall@1000', 'NDCG@10', 'NDCG@1000']
    
    for metric in metrics:
        llm_val = llm_orig[metric]
        bm25_val = bm25_orig[metric]
        improvement = ((llm_val / bm25_val - 1) * 100) if bm25_val > 0 else 0
        print(f"  {metric}: LLM {llm_val:.4f} vs BM25 {bm25_val:.4f} ({improvement:+.1f}%)")
    print()
    
    # Find best and worst performing systems for each method
    print("Best vs Worst LLM Systems:")
    for metric in metrics:
        best_idx = llm_df[metric].idxmax()
        worst_idx = llm_df[metric].idxmin()
        
        print(f"  {metric}:")
        print(f"    Best:  {llm_df.iloc[best_idx]['System']} ({llm_df.iloc[best_idx][metric]:.4f})")
        print(f"    Worst: {llm_df.iloc[worst_idx]['System']} ({llm_df.iloc[worst_idx][metric]:.4f})")
    print()
    
    print("Best vs Worst BM25 Systems:")
    for metric in metrics:
        best_idx = bm25_df[metric].idxmax()
        worst_idx = bm25_df[metric].idxmin()
        
        print(f"  {metric}:")
        print(f"    Best:  {bm25_df.iloc[best_idx]['System']} ({bm25_df.iloc[best_idx][metric]:.4f})")
        print(f"    Worst: {bm25_df.iloc[worst_idx]['System']} ({bm25_df.iloc[worst_idx][metric]:.4f})")
    print()
    
    print("KEY FINDINGS:")
    print("-" * 50)
    print("• LLM retrieval significantly outperforms BM25 across most metrics")
    print("  for original queries, especially in precision-oriented metrics.")
    print()
    print("• BM25 shows much larger performance degradation with LLM-generated")
    print("  queries, suggesting LLM queries may not be optimized for keyword-based")
    print("  sparse retrieval.")
    print()
    print("• Among LLM-generated queries:")
    print("  - GPT-4o-Mini generally performs best in LLM retrieval")
    print("  - Performance gaps are smaller in LLM retrieval vs BM25")
    print("  - All LLM systems show substantial drops in BM25 performance")
    print()
    print("• Recall@1000 shows the smallest relative performance drop in both")
    print("  retrieval methods, indicating LLM queries can find relevant documents")
    print("  but struggle with ranking quality.")
    print()
    print("• The results suggest LLM-generated queries are better suited for")
    print("  neural/dense retrieval than traditional sparse retrieval methods.")

def calculate_correlation_with_performance(correlation_df, llm_df, bm25_df):
    """
    Analyze the relationship between query correlation and ranking performance for both retrieval methods.
    """
    print("=" * 80)
    print("CORRELATION vs PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # Extract correlations with original (first row, skip diagonal)
    correlations = []
    llm_performance_scores = []
    bm25_performance_scores = []
    system_names = []
    
    for i in range(1, len(llm_df)):  # Skip original (index 0)
        system = llm_df.iloc[i]['System']
        
        # Map system names to correlation matrix labels
        correlation_mapping = {
            'Gemini-2.5-Flash-Lite': 'Gemini-Flash-Lite',
            'GPT-4o-Mini': 'GPT-4o-Mini', 
            'GPT-4o-2024-08-06': 'GPT-4o-2024-08-06'
        }
        
        if system in correlation_mapping:
            corr_label = correlation_mapping[system]
            # Get Pearson correlation from your correlation matrix
            correlation_values = {
                'Gemini-Flash-Lite': 0.9372,
                'GPT-4o-Mini': 0.9375,
                'GPT-4o-2024-08-06': 0.9444
            }
            
            if corr_label in correlation_values:
                correlation = correlation_values[corr_label]
                # Use average of all ranking metrics as performance score
                llm_avg_performance = llm_df.iloc[i][['Reciprocal Rank', 'Recall@1000', 'NDCG@10', 'NDCG@1000']].mean()
                bm25_avg_performance = bm25_df.iloc[i][['Reciprocal Rank', 'Recall@1000', 'NDCG@10', 'NDCG@1000']].mean()
                
                correlations.append(correlation)
                llm_performance_scores.append(llm_avg_performance)
                bm25_performance_scores.append(bm25_avg_performance)
                system_names.append(system)
    
    # Create scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4']
    
    # LLM retrieval correlation vs performance
    for i, (corr, perf, name) in enumerate(zip(correlations, llm_performance_scores, system_names)):
        ax1.scatter(corr, perf, s=200, color=colors[i], alpha=0.7, label=name)
        ax1.annotate(name, (corr, perf), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Query Correlation with Original (Pearson r)', fontsize=11)
    ax1.set_ylabel('Average LLM Retrieval Performance', fontsize=11)
    ax1.set_title('Query Correlation vs LLM Retrieval Performance', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # BM25 retrieval correlation vs performance
    for i, (corr, perf, name) in enumerate(zip(correlations, bm25_performance_scores, system_names)):
        ax2.scatter(corr, perf, s=200, color=colors[i], alpha=0.7, label=name)
        ax2.annotate(name, (corr, perf), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Query Correlation with Original (Pearson r)', fontsize=11)
    ax2.set_ylabel('Average BM25 Retrieval Performance', fontsize=11)
    ax2.set_title('Query Correlation vs BM25 Retrieval Performance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('correlation_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate correlations between query correlation and performance
    if len(correlations) >= 3:
        llm_perf_corr = np.corrcoef(correlations, llm_performance_scores)[0, 1]
        bm25_perf_corr = np.corrcoef(correlations, bm25_performance_scores)[0, 1]
        
        print(f"Correlation between query similarity and LLM retrieval performance: r = {llm_perf_corr:.4f}")
        print(f"Correlation between query similarity and BM25 retrieval performance: r = {bm25_perf_corr:.4f}")
        print()
        
        def interpret_correlation(corr, method_name):
            if corr > 0.5:
                return f"Strong positive relationship in {method_name}: Higher query correlation leads to better performance"
            elif corr > 0.2:
                return f"Moderate positive relationship in {method_name}: Some correlation between query similarity and performance"
            elif corr < -0.2:
                return f"Negative relationship in {method_name}: Higher correlation may lead to worse performance"
            else:
                return f"Weak relationship in {method_name}: Query correlation doesn't strongly predict performance"
        
        print(interpret_correlation(llm_perf_corr, "LLM retrieval"))
        print(interpret_correlation(bm25_perf_corr, "BM25 retrieval"))
        print()
        
        print("INTERESTING OBSERVATION:")
        print(f"The correlation-performance relationship differs between retrieval methods.")
        print(f"This suggests that query characteristics that matter for sparse vs dense")
        print(f"retrieval may be different.")

if __name__ == "__main__":
    # Create ranking results data for both retrieval methods
    llm_df, bm25_df = create_ranking_results_data()
    
    # Create visualizations
    print("Creating comprehensive ranking results visualizations...")
    
    # Main comparison plots
    fig1 = plot_ranking_metrics(llm_df, bm25_df)
    
    # Radar charts
    fig2 = plot_performance_radar(llm_df, bm25_df)
    
    # Relative performance
    fig3, llm_relative_df, bm25_relative_df = plot_relative_performance(llm_df, bm25_df)
    
    # Print summary
    print_ranking_summary(llm_df, bm25_df)
    
    # Analyze correlation vs performance (using correlation data from your other script)
    calculate_correlation_with_performance(None, llm_df, bm25_df)
    
    # Save data to CSV
    llm_df.to_csv('outputs/llm_ranking_results.csv', index=False)
    bm25_df.to_csv('outputs/bm25_ranking_results.csv', index=False)
    llm_relative_df.to_csv('outputs/llm_relative_performance.csv', index=False)
    bm25_relative_df.to_csv('outputs/bm25_relative_performance.csv', index=False)
    
    print()
    print("Files saved:")
    print("- ranking_results_comparison.png")
    print("- ranking_results_radar.png") 
    print("- ranking_results_relative.png")
    print("- correlation_vs_performance.png")
    print("- llm_ranking_results.csv")
    print("- bm25_ranking_results.csv")
    print("- llm_relative_performance.csv")
    print("- bm25_relative_performance.csv")
