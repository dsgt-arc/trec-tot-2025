# Create a diagnostic script: check_qrel_coverage.py
import numpy as np
import os
from pathlib import Path

def check_qrel_coverage():
    """Check if qrel documents are in the entertainment index."""
    
    # Load qrels
    qrel_docs = set()
    with open('/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/dev1-2025-qrel.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qrel_docs.add(parts[2])  # corpus_id
    
    # Load entertainment index IDs
    entertainment_ids = np.load('/storage/home/hcoda1/5/rmehta307/scratch/trec-tot-2025/faiss_indexes/ids_entertainment.npy', allow_pickle=True)
    entertainment_ids = set(str(id) for id in entertainment_ids)
    
    # Check coverage
    missing_docs = qrel_docs - entertainment_ids
    coverage = (len(qrel_docs) - len(missing_docs)) / len(qrel_docs) * 100
    
    print(f"QRel documents: {len(qrel_docs)}")
    print(f"Entertainment index size: {len(entertainment_ids)}")
    print(f"Coverage: {coverage:.1f}%")
    print(f"Missing docs: {len(missing_docs)}")
    
    if missing_docs:
        print(f"Missing document IDs: {list(missing_docs)[:10]}...")  # First 10
    
    return coverage, missing_docs

coverage, missing = check_qrel_coverage()