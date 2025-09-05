#!/usr/bin/env python3
"""
Script to identify which batches need to be rerun based on missing queries.
"""

import json
from pathlib import Path


def load_queries(queries_file):
    """Load all queries and create a mapping from query_id to index."""
    queries = []
    with open(queries_file, "r") as f:
        for line in f:
            query = json.loads(line.strip())
            queries.append(str(query["query_id"]))  # Convert to string for consistency

    # Create mapping from query_id to its index in the original file
    query_to_index = {qid: idx for idx, qid in enumerate(queries)}
    return queries, query_to_index


def find_missing_batches(missing_queries_file, queries_file, batch_size=50):
    """Find which batches need to be rerun based on missing queries."""

    # Load all queries
    all_queries, query_to_index = load_queries(queries_file)

    # Load missing queries
    with open(missing_queries_file, "r") as f:
        missing_query_ids = [line.strip() for line in f if line.strip()]

    print(f"Total queries: {len(all_queries)}")
    print(f"Missing queries: {len(missing_query_ids)}")
    print()

    # Map missing queries to their indices
    missing_indices = []
    for qid in missing_query_ids:
        if qid in query_to_index:
            missing_indices.append(query_to_index[qid])
        else:
            print(f"Warning: Query ID {qid} not found in original dataset")

    missing_indices.sort()

    # Find which batches these indices belong to
    batches_needed = set()
    for idx in missing_indices:
        batch_start = (idx // batch_size) * batch_size
        batches_needed.add(batch_start)

    batches_needed = sorted(batches_needed)

    print("Missing query indices and their batches:")
    print("Query Index -> Batch Start")
    print("-" * 30)
    for idx in missing_indices[:20]:  # Show first 20
        batch_start = (idx // batch_size) * batch_size
        print(f"{idx:4d} -> {batch_start:4d}")

    if len(missing_indices) > 20:
        print(f"... and {len(missing_indices) - 20} more")

    print(f"\nBatches that need to be rerun ({len(batches_needed)} total):")
    print("START_QUERY_INDEX values:")
    for batch_start in batches_needed:
        print(batch_start)

    print("\nRerun commands:")
    print("export RETRIEVAL_MODEL=comb_200_gemini_bm25_bge")
    for batch_start in batches_needed:
        print(f'sbatch -J "$RETRIEVAL_MODEL-{batch_start}" rerank.sbatch')

    # Check which result directories exist
    print("\nChecking existing result directories:")
    base_path = (
        Path.home()
        / "trec-tot-2025"
        / ".scratch"
        / "results"
        / "rerank"
        / "v7"
        / "comb_200_gemini_bm25_bge"
    )

    existing_batches = []
    missing_result_batches = []

    for batch_start in batches_needed:
        # The directory pattern from the sbatch script
        dir_pattern = f"test-2025-500-50-{batch_start}"
        result_dir = base_path / dir_pattern
        result_file = result_dir / "rerank-results-reformatted.txt"

        if result_file.exists():
            existing_batches.append(batch_start)
            print(f"✓ {batch_start}: {result_file}")
        else:
            missing_result_batches.append(batch_start)
            print(f"✗ {batch_start}: {result_file} (missing)")

    if missing_result_batches:
        print(
            f"\nBatches that definitely need to be rerun ({len(missing_result_batches)} total):"
        )
        for batch_start in missing_result_batches:
            print(f"START_QUERY_INDEX={batch_start}")

    if existing_batches:
        print(
            f"\nBatches with existing files but missing queries ({len(existing_batches)} total):"
        )
        print("These might have failed partway through or had errors:")
        for batch_start in existing_batches:
            print(f"START_QUERY_INDEX={batch_start}")


if __name__ == "__main__":
    missing_queries_file = "/tmp/real_missing_queries.txt"
    queries_file = str(
        Path.home()
        / "trec-tot-2025"
        / ".scratch"
        / "data"
        / "official"
        / "test-2025-queries.jsonl"
    )

    find_missing_batches(missing_queries_file, queries_file)
