# llm reranking

This directory contains supplementary code for wenxin's reranking code.
I'm setting it up so that it's relatively straightforward to run the code.

First make sure you have a copy of https://zenodo.org/records/15356599 locally.
For reranking I only need the queries and qrels, but I've included everything for completeness.

```bash
rclone sync $HOME/scratch/trec-tot-2025/results/rerank gdrive-trec-tot-2025:data/rerank
```

MODEL=google/gemma-3-12b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=gaunernst/gemma-3-12b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=gaunernst/gemma-3-27b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
