#!/usr/bin/env python3
"""
Test script to verify extract_features_v3 functionality.
"""

import sys
import os

# Add the current directory to sys.path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from select_train_samples import extract_features_v1, extract_features_v2, extract_features_v3

def test_feature_extraction():
    """Test all three feature extraction versions."""
    
    # Sample test data
    doc_id = "799"
    query_id = "query_1"
    
    # Mock retrieval results
    sparse_results = {
        "query_1": {
            "123456": (5, 12.5),  # rank=5, score=12.5
            "789012": (1, 25.0),  # rank=1, score=25.0
            "345678": (10, 8.2),   # rank=10, score=8.2
            "2028": (50, 5.0),      # rank=50, score=5.0
            "290": (99, 7.0)       # rank=99, score=7.0
        }
    }
    
    dense_results = {
        "query_1": {
            "123456": (3, 0.85),  # rank=3, score=0.85
            "789012": (2, 0.92),  # rank=2, score=0.92
            "345678": (8, 0.75),   # rank=8, score=0.75
            "290": (109, 0.66)       # rank=109, score=0.66
        }
    }
    
    print("Testing Feature Extraction Functions")
    print("=" * 50)
    
    # Test v1 features
    print("V1 Features (7 features):")
    v1_features = extract_features_v1(doc_id, query_id, sparse_results, dense_results)
    print(f"  Doc {doc_id}: {v1_features}")
    print(f"  Feature count: {len(v1_features)}")
    
    # Test v2 features  
    print("\nV2 Features (7 normalized features):")
    v2_features = extract_features_v2(doc_id, query_id, sparse_results, dense_results)
    print(f"  Doc {doc_id}: {v2_features}")
    print(f"  Feature count: {len(v2_features)}")
    
    # Test v3 features
    print("\nV3 Features (6 features: sparse/dense scores and ranks + pageview + pagerank):")
    try:
        v3_features = extract_features_v3(doc_id, query_id, sparse_results, dense_results)
        print(f"  Doc {doc_id}: {v3_features}")
        print(f"  Feature count: {len(v3_features)}")
        
        # Display feature breakdown
        print("\nV3 Feature Breakdown:")
        feature_names = [
            "sparse_score", "dense_score", "sparse_rank", "dense_rank",
            "log_pageviews", "pagerank"
        ]
        for i, (name, value) in enumerate(zip(feature_names, v3_features)):
            print(f"  {i+1}. {name}: {value:.6f}")
            
    except Exception as e:
        print(f"  Error loading v3 features: {e}")
        print("  This is expected if pageview/pagerank data files are not available.")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_feature_extraction()
