# Centrality

This module computes various centrality measures on Wikipedia article graphs for reranking purposes in the TREC TOT 2025 task.

## Overview

The workflow consists of three main components:

1. **Graph Merging** (`merge-graph.py`): Combines multiple graph sources into a unified graph
2. **Centrality Computation** (`process.py`): Calculates various centrality measures
3. **Personalized PageRank** (`ppr.py`): Computes query-specific personalized PageRank scores

## Graph Sources

The pipeline processes three types of graphs:

- **article-one-hop**: Direct Wikipedia link graph (weight: 1.0)
- **article-meta-two-hop**: Extended Wikipedia link graph with meta pages (weight: 0.5)
- **bge-m3-knn-k15**: Semantic similarity graph using BGE-M3 embeddings (weight: 0.25)

These are merged into a unified `merged-v2` graph using weighted combination.

## Centrality Measures

The following centrality measures are computed for each graph variant:

- **PageRank**: Standard PageRank algorithm (200 iterations, tolerance 1e-10)
- **HITS**: Hub and authority scores (100 iterations, tolerance 1e-8)
- **Degree Centrality**: In-degree and out-degree centrality using rustworkx
- **Raw Degree**: Simple in-degree and out-degree counts using Polars

## Graph Variants Processed

- `bge-m3-knn-k10`: Semantic graph with k=10 nearest neighbors
- `bge-m3-knn-k15`: Semantic graph with k=15 nearest neighbors
- `merged-v2`: Combined graph from multiple sources

## Usage

### Merge Graphs

```bash
sbatch merge-graph.sbatch
```

### Compute Centrality Measures

```bash
sbatch process.sbatch
```

### Compute Personalized PageRank

```bash
sbatch ppr.sbatch
```

## Output

Results are stored in `~/scratch/trec-tot-2025/data/enwiki/processed/centrality/v2.2/{suffix}/`:

- `pagerank.parquet`: PageRank scores
- `degree_centrality.parquet`: Normalized degree centrality
- `degree.parquet`: Raw degree counts

## Notebooks

- `00-rustworkx.ipynb`: Initial exploration of RustWorkX capabilities
- `01-results.ipynb`: Analysis of computed centrality measures
- `02-rerank.ipynb`: Reranking experiments with centrality scores
- `03-rerank-ppr.ipynb`: Personalized PageRank reranking experiments
