import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_correlation_matrices():
    """
    Create correlation matrices for query correlation analysis.
    
    Sets:
    1. Original queries
    2. Gemini-flash-lite generated queries  
    3. GPT-4o-mini generated queries
    4. First 200 words from Wikipedia pages
    5. Random text 200 words
    """
    
    # Define the labels for our sets
    labels = [
        'TREC Original',
        'Gemini-Flash-Lite', 
        'GPT-4o-Mini',
        'GPT-4o-2024-08-06',
        'Wikipedia 200 words',
        'Random 200 words'
    ]
    
    # Initialize correlation matrices (6x6)
    pearson_matrix = np.eye(6)  # Identity matrix (diagonal = 1.0)
    kendall_matrix = np.eye(6)  # Identity matrix (diagonal = 1.0)
    
    # P-value matrices
    pearson_pvalue_matrix = np.zeros((6, 6))
    kendall_pvalue_matrix = np.zeros((6, 6))
    
    # Fill in the correlation values from your results
    # Set 1 vs Set 2 (Original vs Gemini-flash-lite)
    pearson_matrix[0, 1] = pearson_matrix[1, 0] = 0.9372
    kendall_matrix[0, 1] = kendall_matrix[1, 0] = 0.7771
    pearson_pvalue_matrix[0, 1] = pearson_pvalue_matrix[1, 0] = 0.0000
    kendall_pvalue_matrix[0, 1] = kendall_pvalue_matrix[1, 0] = 0.0000
    
    # Set 1 vs Set 3 (Original vs GPT-4o-mini)
    pearson_matrix[0, 2] = pearson_matrix[2, 0] = 0.9375
    kendall_matrix[0, 2] = kendall_matrix[2, 0] = 0.7772
    pearson_pvalue_matrix[0, 2] = pearson_pvalue_matrix[2, 0] = 0.0000
    kendall_pvalue_matrix[0, 2] = kendall_pvalue_matrix[2, 0] = 0.0000
    
    # Set 1 vs Set 4 (Original vs GPT-4o-2024-08-06)
    pearson_matrix[0, 3] = pearson_matrix[3, 0] = 0.9444
    kendall_matrix[0, 3] = kendall_matrix[3, 0] = 0.7902
    pearson_pvalue_matrix[0, 3] = pearson_pvalue_matrix[3, 0] = 0.0000
    kendall_pvalue_matrix[0, 3] = kendall_pvalue_matrix[3, 0] = 0.0000
    
    # Set 1 vs Set 5 (Original vs Wikipedia 200w)
    pearson_matrix[0, 4] = pearson_matrix[4, 0] = 0.6199
    kendall_matrix[0, 4] = kendall_matrix[4, 0] = 0.4257
    pearson_pvalue_matrix[0, 4] = pearson_pvalue_matrix[4, 0] = 0.0000
    kendall_pvalue_matrix[0, 4] = kendall_pvalue_matrix[4, 0] = 0.0000
    
    # Set 1 vs Set 6 (Original vs Random 200w)
    pearson_matrix[0, 5] = pearson_matrix[5, 0] = 0.0784
    kendall_matrix[0, 5] = kendall_matrix[5, 0] = 0.0516
    pearson_pvalue_matrix[0, 5] = pearson_pvalue_matrix[5, 0] = 0.1251
    kendall_pvalue_matrix[0, 5] = kendall_pvalue_matrix[5, 0] = 0.1313
    
    # Note: We only have correlations with Set 1 (Original) as the reference
    # Other pairwise correlations would need to be calculated separately
    
    # Create DataFrames for easier handling
    pearson_df = pd.DataFrame(pearson_matrix, index=labels, columns=labels)
    kendall_df = pd.DataFrame(kendall_matrix, index=labels, columns=labels)
    pearson_pvalue_df = pd.DataFrame(pearson_pvalue_matrix, index=labels, columns=labels)
    kendall_pvalue_df = pd.DataFrame(kendall_pvalue_matrix, index=labels, columns=labels)
    
    return pearson_df, kendall_df, pearson_pvalue_df, kendall_pvalue_df

def plot_correlation_matrices(pearson_df, kendall_df, pearson_pvalue_df, kendall_pvalue_df):
    """
    Plot correlation matrices with significance indicators.
    """
    
    # Set up the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create custom annotations that include p-values
    def create_annotations(corr_df, pvalue_df):
        annotations = np.empty_like(corr_df, dtype=object)
        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                if i == j:
                    annotations[i, j] = "1.00"
                elif corr_df.iloc[i, j] != 0:  # Only show values we have data for
                    corr_val = corr_df.iloc[i, j]
                    p_val = pvalue_df.iloc[i, j]
                    if p_val < 0.001:
                        sig_indicator = "***"
                    elif p_val < 0.01:
                        sig_indicator = "**"
                    elif p_val < 0.05:
                        sig_indicator = "*"
                    else:
                        sig_indicator = ""
                    annotations[i, j] = f"{corr_val:.3f}{sig_indicator}"
                else:
                    annotations[i, j] = "N/A"
        return annotations
    
    # Pearson correlation matrix
    pearson_annotations = create_annotations(pearson_df, pearson_pvalue_df)
    
    # Mask for values we don't have (only show correlations with Original)
    mask_pearson = np.zeros_like(pearson_df, dtype=bool)
    for i in range(1, 6):
        for j in range(1, 6):
            if i != j:
                mask_pearson[i, j] = True
    
    sns.heatmap(pearson_df, annot=pearson_annotations, fmt='', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, mask=mask_pearson,
                square=True, linewidths=0.5, ax=axes[0])
    axes[0].set_title('Pearson Correlation Matrix\n(*** p<0.001, ** p<0.01, * p<0.05)', 
                      fontsize=12, fontweight='bold')
    
    # Kendall's Tau correlation matrix  
    kendall_annotations = create_annotations(kendall_df, kendall_pvalue_df)
    
    # Same mask for Kendall
    mask_kendall = np.zeros_like(kendall_df, dtype=bool)
    for i in range(1, 6):
        for j in range(1, 6):
            if i != j:
                mask_kendall[i, j] = True
                
    sns.heatmap(kendall_df, annot=kendall_annotations, fmt='', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, mask=mask_kendall,
                square=True, linewidths=0.5, ax=axes[1])
    axes[1].set_title("Kendall's Tau Correlation Matrix\n(*** p<0.001, ** p<0.01, * p<0.05)", 
                      fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('query_correlation_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_correlation_summary(pearson_df, kendall_df, pearson_pvalue_df, kendall_pvalue_df):
    """
    Print a summary of the correlation results.
    """
    print("=" * 80)
    print("QUERY CORRELATION ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    
    print("Dataset Descriptions:")
    print("1. Original: Original TREC queries")
    print("2. Gemini-Flash-Lite: Queries generated by Gemini Flash Lite")  
    print("3. GPT-4o-Mini: Queries generated by GPT-4o-Mini")
    print("4. GPT-4o-2024-08-06: Queries generated by GPT-4o-2024-08-06")
    print("5. Wikipedia 200w: First 200 words from Wikipedia pages")
    print("6. Random 200w: Random text of 200 words")
    print()
    
    print("Key Findings:")
    print("-" * 40)
    
    # Get correlations with original (row 0)
    original_correlations = []
    for i, label in enumerate(pearson_df.columns[1:], 1):
        pearson_corr = pearson_df.iloc[0, i]
        kendall_corr = kendall_df.iloc[0, i]
        pearson_p = pearson_pvalue_df.iloc[0, i]
        kendall_p = kendall_pvalue_df.iloc[0, i]
        
        if pearson_corr != 0:  # Only show correlations we have data for
            original_correlations.append((label, pearson_corr, kendall_corr, pearson_p, kendall_p))
    
    # Sort by Pearson correlation (descending)
    original_correlations.sort(key=lambda x: x[1], reverse=True)
    
    for label, pearson_corr, kendall_corr, pearson_p, kendall_p in original_correlations:
        print(f"Original vs {label}:")
        print(f"  Pearson: r = {pearson_corr:.4f} (p = {pearson_p:.4f})")
        print(f"  Kendall: τ = {kendall_corr:.4f} (p = {kendall_p:.4f})")
        
        if pearson_p < 0.001:
            sig_level = "highly significant"
        elif pearson_p < 0.01:
            sig_level = "very significant" 
        elif pearson_p < 0.05:
            sig_level = "significant"
        else:
            sig_level = "not significant"
            
        print(f"  Statistical significance: {sig_level}")
        print()
    
    print("Interpretation:")
    print("-" * 40)
    print("• All LLM-generated queries (Gemini & GPT-4o variants) show very high")
    print("  correlation with original queries (r > 0.93), suggesting they capture")
    print("  similar semantic content and query characteristics.")
    print()
    print("• GPT-4o-2024-08-06 shows the highest correlation (r = 0.944), slightly")
    print("  outperforming GPT-4o-Mini and Gemini-Flash-Lite.")
    print()
    print("• Wikipedia text shows moderate correlation (r = 0.62), indicating some")
    print("  semantic overlap but different query characteristics.")
    print()
    print("• Random text shows very low correlation (r = 0.08, p > 0.05), confirming")
    print("  the validity of the embedding-based correlation approach.")

if __name__ == "__main__":
    # Create correlation matrices
    pearson_df, kendall_df, pearson_pvalue_df, kendall_pvalue_df = create_correlation_matrices()
    
    # Plot the matrices
    fig = plot_correlation_matrices(pearson_df, kendall_df, pearson_pvalue_df, kendall_pvalue_df)
    
    # Print summary
    print_correlation_summary(pearson_df, kendall_df, pearson_pvalue_df, kendall_pvalue_df)
    
    # Save the correlation data to CSV files for future reference
    pearson_df.to_csv('outputs/pearson_correlation_matrix.csv')
    kendall_df.to_csv('outputs/kendall_correlation_matrix.csv')
    pearson_pvalue_df.to_csv('outputs/pearson_pvalues_matrix.csv')
    kendall_pvalue_df.to_csv('outputs/kendall_pvalues_matrix.csv')

    print()
    print("Files saved:")
    print("- query_correlation_matrices.png")
    print("- pearson_correlation_matrix.csv")
    print("- kendall_correlation_matrix.csv") 
    print("- pearson_pvalues_matrix.csv")
    print("- kendall_pvalues_matrix.csv")
