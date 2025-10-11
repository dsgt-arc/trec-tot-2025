"""
Data Loading Utilities for TREC-ToT 2025 Pipeline

This module handles loading and processing of JSONL query files,
including grouping query variants by query_id.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryData:
    """Represents query data with variants."""
    
    def __init__(self, query_id: str, variants: List[str]):
        self.query_id = query_id
        self.variants = variants  # All variants including original
        self.original_query = variants[0] if variants else ""
        self.relaxed_queries = variants[1:] if len(variants) > 1 else []
    
    def __repr__(self):
        return f"QueryData(id={self.query_id}, variants={len(self.variants)})"


class DataLoader:
    """
    Handles loading and processing of JSONL query files.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_jsonl_queries(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load queries from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of dictionaries with query_id and query fields
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If JSON parsing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Query file not found: {file_path}")
        
        queries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        query_data = json.loads(line)
                        
                        # Validate required fields
                        if 'query_id' not in query_data or 'query' not in query_data:
                            self.logger.warning(
                                f"Line {line_num}: Missing required fields 'query_id' or 'query'"
                            )
                            continue
                        
                        queries.append({
                            'query_id': str(query_data['query_id']),
                            'query': str(query_data['query']).strip()
                        })
                        
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Line {line_num}: JSON decode error: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise
        
        self.logger.info(f"Loaded {len(queries)} queries from {file_path}")
        return queries
    
    def load_json_queries(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load queries from a JSON file (array format).
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of dictionaries with query_id and query fields
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Query file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON file must contain an array of query objects")
            
            queries = []
            for i, query_data in enumerate(data):
                if not isinstance(query_data, dict):
                    self.logger.warning(f"Item {i}: Expected dict, got {type(query_data)}")
                    continue
                
                if 'query_id' not in query_data or 'query' not in query_data:
                    self.logger.warning(f"Item {i}: Missing required fields")
                    continue
                
                queries.append({
                    'query_id': str(query_data['query_id']),
                    'query': str(query_data['query']).strip()
                })
            
            self.logger.info(f"Loaded {len(queries)} queries from {file_path}")
            return queries
            
        except Exception as e:
            self.logger.error(f"Error reading JSON file {file_path}: {e}")
            raise
    
    def group_query_variants(self, queries: List[Dict[str, str]], 
                           expected_variants: int = 6) -> Dict[str, QueryData]:
        """
        Group query variants by query_id.
        
        Args:
            queries: List of query dictionaries
            expected_variants: Expected number of variants per query_id
            
        Returns:
            Dictionary mapping query_id to QueryData objects
        """
        grouped = defaultdict(list)
        
        # Group by query_id
        for query in queries:
            grouped[query['query_id']].append(query['query'])
        
        result = {}
        for query_id, variants in grouped.items():
            if len(variants) != expected_variants:
                self.logger.warning(
                    f"Query {query_id}: Expected {expected_variants} variants, "
                    f"found {len(variants)}"
                )
            
            result[query_id] = QueryData(query_id, variants)
        
        self.logger.info(
            f"Grouped {len(queries)} queries into {len(result)} unique query_ids"
        )
        return result
    
    def load_and_group_queries(self, file_path: str, 
                              expected_variants: int = 6) -> Dict[str, QueryData]:
        """
        Load queries from file and group by query_id.
        
        Args:
            file_path: Path to query file (JSONL or JSON)
            expected_variants: Expected number of variants per query_id
            
        Returns:
            Dictionary mapping query_id to QueryData objects
        """
        file_path = Path(file_path)
        
        # Determine file type and load accordingly
        if file_path.suffix == '.json':
            queries = self.load_json_queries(str(file_path))
        else:
            # Default to JSONL format
            queries = self.load_jsonl_queries(str(file_path))
        
        return self.group_query_variants(queries, expected_variants)


# Convenience function
def load_queries(file_path: str, expected_variants: int = 6) -> Dict[str, QueryData]:
    """
    Convenience function to load and group queries.
    
    Args:
        file_path: Path to query file
        expected_variants: Expected number of variants per query_id
        
    Returns:
        Dictionary mapping query_id to QueryData objects
    """
    loader = DataLoader()
    return loader.load_and_group_queries(file_path, expected_variants)


if __name__ == "__main__":
    # Test the data loader
    sample_file = "/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/relaxed_queries_dev1-2025-queries_sample.jsonl"
    
    try:
        loader = DataLoader()
        grouped_queries = loader.load_and_group_queries(sample_file)
        
        # Show sample query data
        print("\n=== Sample Query Data ===")
        for i, (query_id, query_data) in enumerate(list(grouped_queries.items())[:2]):
            print(f"\nQuery ID: {query_id}")
            print(f"  Variants: {len(query_data.variants)}")
            for j, variant in enumerate(query_data.variants[:2]):
                print(f"    {j+1}: {variant[:80]}...")
        
    except Exception as e:
        print(f"Error testing data loader: {e}")