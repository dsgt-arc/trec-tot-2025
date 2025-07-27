# llm reranking

This directory contains supplementary code for wenxin's reranking code.
I'm setting it up so that it's relatively straightforward to run the code.

First make sure you have a copy of https://zenodo.org/records/15356599 locally.
For reranking I only need the queries and qrels, but I've included everything for completeness.

```bash
rclone sync $HOME/scratch/trec-tot-2025/results/rerank gdrive-trec-tot-2025:data/rerank
```
