#!/usr/bin/env python3
"""
Script to reformat TREC result files for evaluation.

This script processes TREC run files by:
- Converting space-separated fields to tab-separated fields
- Reassigning scores: starts at 5.0 for the first document of each query, 
  decreases by 1.0 for subsequent documents, with a minimum score of 1.0
- Maintaining the original ranking order

Input: Standard TREC run file format (space-separated)
Output: Tab-separated TREC file with reformatted scores

Usage Example:
    python rewrite_score.py rerank-results.txt
    # Output: rerank-results-reformatted.txt
    
    python rewrite_score.py rerank-results.txt -o custom-output.txt
    # Output: custom-output.txt

The reformatted file can be used with trec_eval tool.
"""

import sys
import argparse
from pathlib import Path


def reformat_trec_file(input_file, output_file):
    """
    Reformat TREC result file with tab separators and modified scores.
    
    Args:
        input_file (str): Path to input TREC file
        output_file (str): Path to output TREC file
    """
    current_query = None
    query_row_count = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            # Split the line by spaces (standard TREC format)
            fields = line.split()
            
            if len(fields) != 6:
                print(f"Warning: Unexpected line format: {line}")
                continue
                
            query_id, q0, doc_id, rank, score, run_name = fields
            
            # Check if this is a new query
            if current_query != query_id:
                current_query = query_id
                query_row_count = 0

            # Calculate score: start at 5.0, decrease by 1 for each row, minimum 1.0
            new_score = max(5.0 - query_row_count, 1.0)
            query_row_count += 1

            # Write tab-separated output
            output_line = f"{query_id} {q0} {doc_id} {rank} {new_score} {run_name}"
            outfile.write(output_line + "\n")

def main():
    parser = argparse.ArgumentParser(description="Reformat TREC result files")
    parser.add_argument("input_file", help="Input TREC result file")
    parser.add_argument("-o", "--output", help="Output file (default: input_file with -reformatted.txt)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        if input_path.suffix.lower() == '.txt':
            # Remove .txt and add -reformatted.txt
            base_name = input_path.stem  # filename without extension
            output_path = input_path.parent / f"{base_name}-reformatted.txt"
        else:
            # Add -reformatted.txt to the existing filename
            output_path = input_path.parent / f"{input_path.name}-reformatted.txt"

    print(f"Reformatting {input_path} -> {output_path}")
    reformat_trec_file(input_path, output_path)
    print(f"Successfully reformatted file. Output saved to: {output_path}")

if __name__ == "__main__":
    main()
