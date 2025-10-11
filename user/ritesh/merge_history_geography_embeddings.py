#!/usr/bin/env python3
"""
Script to merge the two history_geography topic BGE-M3 embedding parquet files into one.
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def merge_history_geography_embeddings():
    """
    Merge the two history_geography embedding files into one combined file.
    """
    # Define paths
    input_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/bge-m3-embeddings_shards_from_parquet_cleaned"
    output_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/bge-m3-embeddings_shards_from_parquet_cleaned"
    
    # Input files
    file1 = os.path.join(input_dir, "history_geography_cleaned_half1_emb_bge-m3.parquet")
    file2 = os.path.join(input_dir, "history_geography_cleaned_half2_emb_bge-m3.parquet")
    
    # Output file
    output_file = os.path.join(output_dir, "history_geography_cleaned_emb_bge-m3.parquet")
    
    # Check if input files exist
    if not os.path.exists(file1):
        print(f"Error: {file1} not found!")
        return
    if not os.path.exists(file2):
        print(f"Error: {file2} not found!")
        return
    
    print(f"Loading first file: {file1}")
    df1 = pd.read_parquet(file1)
    print(f"First file shape: {df1.shape}")
    
    print(f"Loading second file: {file2}")
    df2 = pd.read_parquet(file2)
    print(f"Second file shape: {df2.shape}")
    
    # Concatenate the dataframes
    print("Merging dataframes...")
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined shape: {combined_df.shape}")
    
    # Verify the structure
    print(f"Columns: {combined_df.columns.tolist()}")
    print(f"Sample IDs: {combined_df['id'].head().tolist()}")
    
    # Save the combined file
    print(f"Saving combined file to: {output_file}")
    combined_df.to_parquet(output_file, compression="zstd", index=False)
    
    # Verify the output file
    print("Verifying output file...")
    verification_df = pd.read_parquet(output_file)
    print(f"Output file shape: {verification_df.shape}")
    
    # Clean up memory
    del df1, df2, combined_df, verification_df
    
    print("Merge completed successfully!")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    merge_history_geography_embeddings()
