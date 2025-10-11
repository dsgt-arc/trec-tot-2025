#!/usr/bin/env python3
"""
Script to sort Wikipedia articles by Wikipedia ID and generate TREC queries.
Reads from outputs/sampled_wikipedia_articles.csv and creates a new JSONL file with query_id, wikipedia_id, title, infobox_type, text.
The text field is loaded from the corpus using the offset mapping file.
Query IDs start from 60000.
"""

import pandas as pd
import os
import json

def load_corpus_text(wikipedia_id, offset_mapping, corpus_file_path):
    """
    Load the text for a given Wikipedia ID using the offset mapping.
    
    Args:
        wikipedia_id: The Wikipedia ID as a string
        offset_mapping: Dictionary with offset information
        corpus_file_path: Path to the corpus JSONL file
    
    Returns:
        The text content or None if not found
    """
    if str(wikipedia_id) not in offset_mapping:
        print(f"Warning: Wikipedia ID {wikipedia_id} not found in offset mapping")
        return None
    
    offsets = offset_mapping[str(wikipedia_id)]
    offset_start = offsets["offset_start"]
    offset_end = offsets["offset_end"]
    
    try:
        with open(corpus_file_path, 'r', encoding='utf-8') as f:
            f.seek(offset_start)
            text_chunk = f.read(offset_end - offset_start + 1)
            
            # Parse the JSONL line
            line_data = json.loads(text_chunk.strip())
            return line_data.get('text', '')
    except Exception as e:
        print(f"Error loading text for Wikipedia ID {wikipedia_id}: {e}")
        return None

def main():
    # Define input and output file paths
    input_file = "outputs/sampled_wikipedia_articles.csv"
    output_file = "outputs/trec_queries_sorted.jsonl"
    
    # Paths to corpus data files
    offset_mapping_file = "/home/wenxin/project/data/2025/corpus-offset-mapping.json"
    corpus_file = "/home/wenxin/project/data/2025/corpus.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    # Load the corpus offset mapping
    print("Loading corpus offset mapping...")
    try:
        with open(offset_mapping_file, 'r', encoding='utf-8') as f:
            offset_mapping = json.load(f)
        print(f"Loaded offset mapping for {len(offset_mapping)} articles")
    except Exception as e:
        print(f"Error loading offset mapping: {e}")
        return
    
    try:
        # Read the CSV file
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        
        # Display basic info about the data
        print(f"Found {len(df)} articles")
        print(f"Columns: {list(df.columns)}")
        
        # Sort by wikipedia_id
        print("Sorting by wikipedia_id...")
        df_sorted = df.sort_values('wikipedia_id').reset_index(drop=True)
        
        # Add query_id starting from 60000
        df_sorted['query_id'] = range(60000, 60000 + len(df_sorted))
        
        # Add text field by loading from corpus
        print("Loading text content from corpus...")
        df_sorted['text'] = df_sorted['wikipedia_id'].apply(
            lambda wiki_id: load_corpus_text(wiki_id, offset_mapping, corpus_file)
        )
        
        # Check how many texts were successfully loaded
        texts_loaded = df_sorted['text'].notna().sum()
        print(f"Successfully loaded text for {texts_loaded}/{len(df_sorted)} articles")
        
        # Reorder columns to include text: query_id, wikipedia_id, title, infobox_type, text
        df_output = df_sorted[['query_id', 'wikipedia_id', 'title', 'infobox_type', 'text']]
        
        # Save to output file in JSONL format
        print(f"Saving sorted data to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in df_output.iterrows():
                record = {
                    'query_id': int(row['query_id']),
                    'wikipedia_id': int(row['wikipedia_id']),
                    'title': row['title'],
                    'infobox_type': row['infobox_type'],
                    'text': row['text']
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # Display summary
        print(f"Successfully created {output_file}")
        print(f"Output contains {len(df_output)} rows")
        print(f"Columns: {list(df_output.columns)}")
        print("\nFirst few rows of output:")
        print(df_output.head())
        
        print(f"\nQuery ID range: {df_output['query_id'].min()} to {df_output['query_id'].max()}")
        print(f"Wikipedia ID range: {df_output['wikipedia_id'].min()} to {df_output['wikipedia_id'].max()}")
        
        # Show text statistics
        text_lengths = df_output['text'].str.len()
        print(f"\nText statistics:")
        print(f"Articles with text: {df_output['text'].notna().sum()}")
        print(f"Articles without text: {df_output['text'].isna().sum()}")
        if text_lengths.notna().any():
            print(f"Average text length: {text_lengths.mean():.0f} characters")
            print(f"Text length range: {text_lengths.min():.0f} to {text_lengths.max():.0f} characters")
        
    except Exception as e:
        print(f"Error processing the file: {e}")

if __name__ == "__main__":
    main()
