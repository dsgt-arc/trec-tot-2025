#!/usr/bin/env python3
"""
Script to generate queries.jsonl file from entity classification results.
Supports two modes:
1. Document mode: Read query_id and entity_id from classification TSV file,
   look up document text in corpus using offset mapping, and use first 200 words
   of document text as query
2. Random mode: Generate queries with random text of specified word count

Generates queries.jsonl with query_id and query fields.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re
import random

def get_document_content(doc_id: str, corpus_file: str,
                         offset_map: Dict[str, Dict[str, int]]) -> Dict[str, str]:
    """Get document content from corpus file using byte offsets."""
    if doc_id not in offset_map:
        return {"title": "", "text": ""}
    
    offset_start = offset_map[doc_id]["offset_start"]
    offset_end = offset_map[doc_id]["offset_end"]
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        f.seek(offset_start)
        content = f.read(offset_end - offset_start)
        
        # Parse the JSON content
        doc_data = json.loads(content.strip())
        return {
            "title": doc_data.get('title', ''),
            "text": doc_data.get('text', '')
        }
    return {"title": "", "text": ""}

def load_corpus_offset(offset_file: str) -> Dict[str, Dict[str, int]]:
    """Load corpus offset mapping."""
    with open(offset_file, 'r', encoding='utf-8') as f:
        offset_map = json.load(f)
    return offset_map

def read_classification_tsv(tsv_file: str) -> List[Tuple[str, str]]:
    """Read classification TSV file and return list of (query_id, entity_id) pairs."""
    results = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query_id = parts[0]
                entity_id = parts[1]
                results.append((query_id, entity_id))
    return results

def extract_first_n_words(text: str, n: int = 200) -> str:
    """Extract first n words from text."""
    words = text.split()
    if len(words) <= n:
        return text
    return ' '.join(words[:n])

def generate_random_text(n_words: int = 200) -> str:
    """Generate random text with approximately n words."""
    # Common English words for generating random text
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him",
        "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only",
        "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want",
        "because", "any", "these", "give", "day", "most", "us", "is", "water", "long", "find", "here", "thing", "great", "man", "world", "life",
        "still", "public", "human", "read", "left", "study", "those", "both", "group", "often", "general", "level", "order", "open", "young",
        "write", "program", "follow", "around", "house", "plant", "grow", "keep", "start", "thought", "school", "story", "example", "paper",
        "play", "run", "move", "right", "boy", "old", "too", "same", "tell", "does", "set", "three", "state", "never", "become", "between",
        "high", "really", "something", "most", "another", "much", "family", "own", "leave", "put", "old", "while", "mean", "on", "keep", "student",
        "why", "let", "great", "same", "big", "group", "begin", "seem", "country", "help", "talk", "where", "turn", "problem", "every", "start",
        "hand", "might", "american", "show", "part", "about", "against", "place", "over", "such", "again", "few", "case", "most", "week", "company",
        "where", "system", "each", "right", "program", "hear", "so", "question", "during", "work", "play", "government", "run", "small", "number",
        "off", "always", "move", "like", "night", "live", "point", "believe", "hold", "today", "bring", "happen", "next", "without", "before",
        "large", "all", "million", "must", "home", "under", "water", "room", "write", "mother", "area", "national", "money", "story", "young"
    ]
    
    # Generate random words
    words = random.choices(common_words, k=n_words)
    return ' '.join(words)

def main():
    parser = argparse.ArgumentParser(description="Generate queries.jsonl from entity classification results")
    parser.add_argument("--classification_file", 
                       default="outputs/classification/dev3_first_100_entity_classification.tsv",
                       help="Path to classification TSV file")
    parser.add_argument("--data_path", 
                       default="/home/wenxin/project/data/2025",
                       help="Path to data directory")
    parser.add_argument("--output_file", 
                       default="outputs/random_200_queries.jsonl",
                       help="Output queries JSONL file")
    parser.add_argument("--word_limit", 
                       type=int, 
                       default=200,
                       help="Number of words to extract from document text")
    parser.add_argument("--query_mode", 
                       choices=["document", "random"],
                       default="document",
                       help="Query generation mode: 'document' uses document text, 'random' generates random text")
    
    args = parser.parse_args()
    
    # File paths
    corpus_file = f"{args.data_path}/corpus.jsonl"
    offset_file = f"{args.data_path}/corpus-offset-mapping.json"
    
    # Check if required files exist
    if not Path(args.classification_file).exists():
        print(f"Error: Classification file not found: {args.classification_file}")
        sys.exit(1)
    
    # Only check corpus files if using document mode
    if args.query_mode == "document":
        if not Path(corpus_file).exists():
            print(f"Error: Corpus file not found: {corpus_file}")
            sys.exit(1)
            
        if not Path(offset_file).exists():
            print(f"Error: Offset mapping file not found: {offset_file}")
            sys.exit(1)
    
    print(f"Step 1: Reading classification results from {args.classification_file}...")
    classification_data = read_classification_tsv(args.classification_file)
    print(f"Loaded {len(classification_data)} query-entity pairs")
    
    # Only load corpus data if using document mode
    if args.query_mode == "document":
        print(f"Step 2: Loading corpus offset mapping from {offset_file}...")
        offset_map = load_corpus_offset(offset_file)
        print(f"Loaded {len(offset_map)} offset mappings")
    else:
        offset_map = {}
        print(f"Step 2: Using random text mode, skipping corpus data loading")
    
    print(f"Step 3: Generating queries using {args.query_mode} mode...")
    
    # Set random seed for reproducible results when using random mode
    if args.query_mode == "random":
        random.seed(42)
    queries_data = []
    missing_entities = []
    
    for i, (query_id, entity_id) in enumerate(classification_data):
        print(f"Processing {i+1}/{len(classification_data)}: query_id={query_id}, entity_id={entity_id}")
        
        try:
            if args.query_mode == "document":
                # Get document content
                doc_content = get_document_content(entity_id, corpus_file, offset_map)
                
                if not doc_content["text"]:
                    print(f"  Warning: No text found for entity {entity_id}")
                    missing_entities.append(entity_id)
                    continue
                
                # Extract first N words from document text
                query_text = extract_first_n_words(doc_content["text"], args.word_limit)
                
                if not query_text.strip():
                    print(f"  Warning: Empty query text for entity {entity_id}")
                    missing_entities.append(entity_id)
                    continue
                    
            elif args.query_mode == "random":
                # Generate random text
                query_text = generate_random_text(args.word_limit)
            
            # Add to queries data
            queries_data.append({
                "query_id": query_id,
                "query": query_text.strip()
            })
            
            print(f"  Generated query with {len(query_text.split())} words")
            
        except Exception as e:
            print(f"  Error processing entity {entity_id}: {e}")
            missing_entities.append(entity_id)
    
    print(f"Step 4: Writing {len(queries_data)} queries to {args.output_file}...")
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write queries to JSONL file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for query_data in queries_data:
            f.write(json.dumps(query_data, ensure_ascii=False) + '\n')
    
    print(f"Completed! Generated {len(queries_data)} queries successfully.")
    
    if missing_entities:
        print(f"Warning: {len(missing_entities)} entities were not processed:")
        for entity_id in missing_entities[:10]:  # Show first 10
            print(f"  {entity_id}")
        if len(missing_entities) > 10:
            print(f"  ... and {len(missing_entities) - 10} more")
    
    # Print some statistics
    if queries_data:
        word_counts = [len(q["query"].split()) for q in queries_data]
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        
        print(f"\nQuery statistics:")
        print(f"  Average words per query: {avg_words:.1f}")
        print(f"  Min words: {min_words}")
        print(f"  Max words: {max_words}")
        
        # Show first few queries as examples
        print(f"\nFirst 3 generated queries:")
        for i, query_data in enumerate(queries_data[:3]):
            print(f"  Query {i+1} (ID: {query_data['query_id']}):")
            print(f"    {query_data['query'][:100]}{'...' if len(query_data['query']) > 100 else ''}")

if __name__ == "__main__":
    main()
