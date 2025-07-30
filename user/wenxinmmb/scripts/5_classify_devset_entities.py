#!/usr/bin/env python3
"""
Script to classify first 100 queries from dev3 set.
1. Extract qrel entity IDs
2. Use corpus offset mapping to find document offsets  
3. Extract documents from corpus-first-1000.jsonl
4. Classify documents as film, person, or place
5. Write results to TSV file
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re

def read_qrel_file(qrel_path: str, limit: int = 100) -> List[Tuple[str, str]]:
    """Read qrel file and return list of (query_id, entity_id) pairs."""
    qrel_data = []
    with open(qrel_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = parts[0]
                entity_id = parts[2]  # Third column is the entity ID
                qrel_data.append((query_id, entity_id))
    return qrel_data

def load_corpus_offset_mapping(mapping_path: str) -> Dict[str, int]:
    """Load corpus offset mapping from JSON file."""
    print(f"Loading corpus offset mapping from {mapping_path}...")
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    print(f"Loaded {len(mapping)} offset mappings")
    return mapping

def extract_document_at_offset(corpus_path: str, offset: int) -> Dict:
    """Extract a single document from the corpus at the given offset."""
    with open(corpus_path, 'r') as f:
        f.seek(offset)
        line = f.readline()
        return json.loads(line.strip())

def classify_document_content(doc: Dict) -> str:
    """
    Classify document as 'film', 'person', or 'place' based on text content.
    Uses simple keyword-based classification.
    """
    text = doc.get('text', '').lower()
    title = doc.get('title', '').lower()
    content = text + ' ' + title
    
    # Film indicators
    film_keywords = [
        'film', 'movie', 'cinema', 'directed by', 'starring', 'cast', 
        'box office', 'screenplay', 'producer', 'cinematography',
        'released', 'premiere', 'genre', 'plot', 'actor', 'actress',
        'character', 'scene', 'sequel', 'franchise', 'adaptation',
        'hollywood', 'studio', 'distribution', 'critic', 'review',
        'oscar', 'academy award', 'golden globe', 'cannes'
    ]
    
    # Person indicators  
    person_keywords = [
        'born', 'died', 'birth', 'death', 'age', 'married', 'spouse',
        'education', 'career', 'profession', 'occupation', 'biography',
        'childhood', 'family', 'parents', 'children', 'siblings',
        'graduated', 'studied', 'worked', 'employed', 'founded',
        'established', 'created', 'invented', 'discovered', 'wrote',
        'published', 'authored', 'composed', 'painted', 'designed',
        'performed', 'sang', 'acted', 'played', 'coached', 'taught'
    ]
    
    # Place indicators
    place_keywords = [
        'located', 'situated', 'geography', 'population', 'area',
        'square miles', 'square kilometers', 'city', 'town', 'village',
        'county', 'state', 'country', 'nation', 'continent', 'region',
        'capital', 'municipality', 'district', 'province', 'territory',
        'climate', 'weather', 'temperature', 'rainfall', 'mountains',
        'river', 'lake', 'ocean', 'sea', 'coast', 'border', 'boundary',
        'landmark', 'attraction', 'tourism', 'economy', 'industry',
        'government', 'mayor', 'governor', 'council', 'residents',
        'inhabitants', 'demographics', 'ethnic', 'language', 'culture'
    ]
    
    # Count keyword matches
    film_score = sum(1 for keyword in film_keywords if keyword in content)
    person_score = sum(1 for keyword in person_keywords if keyword in content)
    place_score = sum(1 for keyword in place_keywords if keyword in content)
    
    # Additional heuristics
    
    # Check for birth/death patterns (strong person indicator)
    birth_death_pattern = r'\b(born|died|birth|death)\s+\d{4}\b|\b\d{4}\s*[-–]\s*\d{4}\b|\b\(\d{4}\s*[-–]\s*\d{4}\)'
    if re.search(birth_death_pattern, content):
        person_score += 3
    
    # Check for film year patterns
    film_year_pattern = r'\b(film|movie)\b.*\b\d{4}\b|\b\d{4}\s+(film|movie)\b'
    if re.search(film_year_pattern, content):
        film_score += 2
    
    # Check for location patterns  
    location_pattern = r'\bis\s+(located|situated)\s+in\b|\bpopulation\s+of\s+\d+'
    if re.search(location_pattern, content):
        place_score += 2
    
    # Determine category based on highest score
    scores = {'film': film_score, 'person': person_score, 'place': place_score}
    max_category = max(scores, key=scores.get)
    max_score = scores[max_category]
    
    # If no clear winner or very low scores, try additional checks
    if max_score == 0 or (max_score < 3 and sum(scores.values()) < 5):
        # Check title for obvious indicators
        if any(word in title for word in ['film', 'movie']):
            return 'film'
        if any(word in title for word in ['city', 'town', 'county', 'state', 'country']):
            return 'place'
        # Default fallback - could be improved with more sophisticated NLP
        return 'person'  # Most articles are about people
    
    return max_category

def main():
    split = 'dev2'
    parser = argparse.ArgumentParser(description='Classify dev1 entities as film, person, or place')
    parser.add_argument('--qrel-path', 
                        default=f'/home/wenxin/project/data/2025/{split}-2025/qrel.txt',
                        help='Path to qrel file')
    parser.add_argument('--corpus-path',
                        default='/home/wenxin/project/data/2025/corpus.jsonl', 
                        help='Path to corpus JSONL file')
    parser.add_argument('--mapping-path',
                        default='/home/wenxin/project/data/2025/corpus-offset-mapping.json',
                        help='Path to corpus offset mapping JSON file')
    parser.add_argument('--output-path',
                        default=f'outputs/{split}_entity_classification.tsv',
                        help='Output TSV file path')
    parser.add_argument('--limit', type=int, default=100,
                        help='Number of queries to process')
    
    args = parser.parse_args()
    
    # Verify input files exist
    for path in [args.qrel_path, args.corpus_path, args.mapping_path]:
        if not Path(path).exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
    
    print(f"Processing first {args.limit} queries from dev3 set...")
    
    # Step 1: Read qrel file
    print("Step 1: Reading qrel file...")
    qrel_data = read_qrel_file(args.qrel_path, args.limit)
    print(f"Found {len(qrel_data)} query-entity pairs")
    
    # Step 2: Load corpus offset mapping
    print("Step 2: Loading corpus offset mapping...")
    offset_mapping = load_corpus_offset_mapping(args.mapping_path)
    
    # Step 3: Process each entity
    print("Step 3: Processing entities...")
    results = []
    missing_entities = []
    
    for i, (query_id, entity_id) in enumerate(qrel_data):
        print(f"Processing {i+1}/{len(qrel_data)}: Query {query_id}, Entity {entity_id}")
        
        # Check if entity exists in offset mapping
        if entity_id not in offset_mapping:
            print(f"  Warning: Entity {entity_id} not found in offset mapping")
            missing_entities.append(entity_id)
            continue
        
        # Get document offset and extract document
        offset = offset_mapping[entity_id]['offset_start']
        try:
            doc = extract_document_at_offset(args.corpus_path, offset)
            
            # Verify document ID matches
            doc_id = doc.get('id', '')
            assert doc_id == entity_id, f"Document ID mismatch: expected {entity_id}, got {doc_id}"
            
            # Classify document
            category = classify_document_content(doc)
            title = doc.get('title', 'Unknown')
            url = doc.get('url', 'Unknown')
            
            results.append({
                'query_id': query_id,
                'entity_id': entity_id,
                'title': title,
                'category': category,
                'url': url
            })
            
            print(f"  Title: {title}")
            print(f"  URL: {url}")
            print(f"  Category: {category}")
            
        except Exception as e:
            print(f"  Error processing entity {entity_id}: {e}")
            missing_entities.append(entity_id)
    
    # Step 4: Write results to TSV
    print(f"Step 4: Writing results to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        # Write header
        f.write("query_id\tentity_id\ttitle\tcategory\turl\n")
        
        # Write data
        for result in results:
            f.write(f"{result['query_id']}\t{result['entity_id']}\t{result['title']}\t{result['category']}\t{result['url']}\n")
    
    print(f"Completed! Processed {len(results)} entities successfully.")
    if missing_entities:
        print(f"Warning: {len(missing_entities)} entities were not found in the corpus:")
        for entity_id in missing_entities[:10]:  # Show first 10
            print(f"  {entity_id}")
        if len(missing_entities) > 10:
            print(f"  ... and {len(missing_entities) - 10} more")
    
    # Print category distribution
    categories = [r['category'] for r in results]
    category_counts = {cat: categories.count(cat) for cat in set(categories)}
    print(f"\nCategory distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

if __name__ == "__main__":
    main()
