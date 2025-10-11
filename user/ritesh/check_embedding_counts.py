#!/usr/bin/env python3
"""
Script to check the number of rows in each BGE-M3 embedding parquet file for all topics.
"""

import os
import pandas as pd
import glob
from pathlib import Path


def check_embedding_counts():
    """
    Check the number of rows in each BGE-M3 embedding parquet file.
    """
    # Define the embeddings directory
    embeddings_dir = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/bge-m3-embeddings_shards_from_parquet_cleaned"
    
    # Get all parquet files in the directory
    parquet_files = glob.glob(os.path.join(embeddings_dir, "*_emb_bge-m3.parquet"))
    
    print(f"Found {len(parquet_files)} embedding files:")
    print("=" * 80)
    
    # Sort files for consistent output
    parquet_files.sort()
    
    total_rows = 0
    file_info = []
    
    for file_path in parquet_files:
        try:
            # Get file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            
            # Get filename without path
            filename = os.path.basename(file_path)
            
            # Extract topic name from filename
            topic_name = filename.replace("_cleaned_emb_bge-m3.parquet", "").replace("_cleaned_half1_emb_bge-m3.parquet", "").replace("_cleaned_half2_emb_bge-m3.parquet", "")
            
            # Read parquet file to get row count
            df = pd.read_parquet(file_path)
            row_count = len(df)
            
            # Get column info
            columns = list(df.columns)
            
            # Check if embeddings column exists and get its shape
            embedding_info = ""
            if 'embedding' in df.columns:
                sample_embedding = df['embedding'].iloc[0]
                if isinstance(sample_embedding, list):
                    embedding_dim = len(sample_embedding)
                    embedding_info = f"embedding_dim={embedding_dim}"
                else:
                    embedding_info = f"embedding_type={type(sample_embedding)}"
            
            print(f"Topic: {topic_name:25} | Rows: {row_count:8,} | Size: {file_size:6.1f}MB | {embedding_info}")
            
            total_rows += row_count
            file_info.append({
                'topic': topic_name,
                'filename': filename,
                'rows': row_count,
                'size_mb': file_size,
                'columns': columns,
                'embedding_info': embedding_info
            })
            
            # Clean up memory
            del df
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print("=" * 80)
    print(f"Total rows across all files: {total_rows:,}")
    print(f"Total files processed: {len(file_info)}")
    
    # Summary by topic (handling duplicates)
    print("\nSummary by topic:")
    print("-" * 50)
    topic_summary = {}
    for info in file_info:
        topic = info['topic']
        if topic not in topic_summary:
            topic_summary[topic] = {'rows': 0, 'files': 0, 'size_mb': 0}
        topic_summary[topic]['rows'] += info['rows']
        topic_summary[topic]['files'] += 1
        topic_summary[topic]['size_mb'] += info['size_mb']
    
    for topic, stats in sorted(topic_summary.items()):
        print(f"{topic:25} | Rows: {stats['rows']:8,} | Files: {stats['files']} | Size: {stats['size_mb']:6.1f}MB")
    
    return file_info


if __name__ == "__main__":
    check_embedding_counts()
