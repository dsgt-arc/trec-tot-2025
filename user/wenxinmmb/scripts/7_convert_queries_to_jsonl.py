#!/usr/bin/env python3
"""
Script to convert queries.tsv file to TREC query JSONL format.

This script reads a TSV file with columns QuestionBody, wikipediaURL, totObj, entityId, queryId
and converts it to JSONL format where each line contains query_id and query fields.

Usage:
    python 7_convert_queries_to_jsonl.py <input_tsv_file> <output_jsonl_file>
    
Example:
    python 7_convert_queries_to_jsonl.py queries.tsv queries.jsonl
"""

import csv
import json
import sys
import os
from pathlib import Path


def convert_tsv_to_jsonl(input_tsv_path: str, output_jsonl_path: str) -> None:
    """
    Convert TSV file to JSONL format for TREC queries.
    
    Args:
        input_tsv_path: Path to input TSV file
        output_jsonl_path: Path to output JSONL file
    """
    
    # Check if input file exists
    if not os.path.exists(input_tsv_path):
        print(f"Error: Input file '{input_tsv_path}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    queries_converted = 0
    
    try:
        with open(input_tsv_path, 'r', encoding='utf-8') as tsv_file, \
             open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
            
            # Read TSV file
            reader = csv.DictReader(tsv_file, delimiter='\t')
            
            # Check if required columns exist
            required_columns = ['QuestionBody', 'queryId']
            if not all(col in reader.fieldnames for col in required_columns):
                print(f"Error: Required columns {required_columns} not found in TSV file.")
                print(f"Available columns: {reader.fieldnames}")
                sys.exit(1)
            
            # Convert each row
            for row in reader:
                query_data = {
                    'query_id': row['queryId'],
                    'query': row['QuestionBody']
                }
                
                # Write to JSONL file
                jsonl_file.write(json.dumps(query_data, ensure_ascii=False) + '\n')
                queries_converted += 1
        
        print(f"Successfully converted {queries_converted} queries from '{input_tsv_path}' to '{output_jsonl_path}'")
        
    except FileNotFoundError:
        print(f"Error: File '{input_tsv_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing files: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and execute conversion."""
    
    if len(sys.argv) != 3:
        print("Usage: python 7_convert_queries_to_jsonl.py <input_tsv_file> <output_jsonl_file>")
        print("\nExample:")
        print("  python 7_convert_queries_to_jsonl.py queries.tsv queries.jsonl")
        sys.exit(1)
    
    input_tsv_path = sys.argv[1]
    output_jsonl_path = sys.argv[2]
    
    print(f"Converting '{input_tsv_path}' to '{output_jsonl_path}'...")
    convert_tsv_to_jsonl(input_tsv_path, output_jsonl_path)


if __name__ == "__main__":
    main()
