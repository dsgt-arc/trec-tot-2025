#!/usr/bin/env python3
"""
Script to reformat TREC result files.
- Converts space-separated fields to tab-separated fields
- Sets score to 5.0 for the first row of each query, descreases by 1.0 for subsequent rows, minimum score of 1.0
- Outputs to a new file with ".reformatted" suffix
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
            output_line = f"{query_id}\t{q0}\t{doc_id}\t{rank}\t{new_score}\t{run_name}"
            outfile.write(output_line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Reformat TREC result files")
    parser.add_argument("input_file", help="Input TREC result file")
    parser.add_argument("-o", "--output", help="Output file (default: input_file with .reformatted extension)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(input_path.suffix + ".reformatted")
    
    print(f"Reformatting {input_path} -> {output_path}")
    
    try:
        reformat_trec_file(input_path, output_path)
        print(f"Successfully reformatted file. Output saved to: {output_path}")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # If no command line arguments, use default file names
    if len(sys.argv) == 1:
        dir = "outputs/gemini-gemma-12B"
        input_file = f"{dir}/rerank-results-0-100.txt"
        output_file = f"{dir}/rerank-results-0-100-reformatted-v3.txt"

        if Path(input_file).exists():
            print(f"No arguments provided. Using default: {input_file} -> {output_file}")
            reformat_trec_file(input_file, output_file)
            print(f"Successfully reformatted file. Output saved to: {output_file}")
        else:
            print(f"Default input file '{input_file}' not found.")
            print("Usage: python rewrite_trec_out.py <input_file> [-o output_file]")
            print("   or: place 'rerank-results-0-5.txt' in the same directory")
    else:
        main()
