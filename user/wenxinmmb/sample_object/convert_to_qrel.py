#!/usr/bin/env python3
"""
Script to convert LLM queries JSONL file to TREC qrel format.

The qrel format is: query_id 0 document_id 1
Where:
- query_id: The query identifier from the JSONL
- 0: Iteration number (always 0 by convention)
- document_id: The Wikipedia ID (entity_id from original data)
- 1: Relevance score (always 1 for our ground truth)

Usage:
python convert_to_qrel.py
"""

import json
import sys
import os

def main():
    """
    Main function to convert LLM queries to qrel format.
    """
    # File paths
    llm_queries_file = "outputs/llm-queries-set-1.jsonl"
    output_file = "outputs/qrel-set-1.txt"
    
    # Check if input files exist
    if not os.path.exists(llm_queries_file):
        print(f"Error: LLM queries file not found: {llm_queries_file}")
        sys.exit(1)
    
    print("Converting LLM queries JSONL to TREC qrel format...")
    print(f"Input LLM queries: {llm_queries_file}")
    print(f"Output qrel file: {output_file}")
    
    # Process LLM queries and write qrel entries in the same loop
    print(f"Processing LLM queries from: {llm_queries_file}")
    print(f"Writing qrel file to: {output_file}")
    
    processed_count = 0
    missing_data_count = 0
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(llm_queries_file, 'r', encoding='utf-8') as input_f, \
             open(output_file, 'w', encoding='utf-8') as output_f:
            
            for line in input_f:
                if line.strip():
                    data = json.loads(line.strip())
                    query_id = str(data.get('query_id', ''))
                    wikipedia_id = str(data.get('wikipedia_id', ''))
                    
                    if query_id and wikipedia_id:
                        # Format: query_id 0 document_id 1
                        qrel_entry = f"{query_id} 0 {wikipedia_id} 1"
                        output_f.write(qrel_entry + '\n')
                        processed_count += 1
                    else:
                        print(f"Warning: Missing data for entry - query_id: {query_id}, wikipedia_id: {wikipedia_id}")
                        missing_data_count += 1
    
    except Exception as e:
        print(f"Error processing files: {e}")
        sys.exit(1)
    
    # Summary
    print(f"\n=== Conversion Complete ===")
    print(f"Processed queries: {processed_count}")
    print(f"Missing data entries: {missing_data_count}")
    print(f"Total qrel entries written: {processed_count}")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main()
