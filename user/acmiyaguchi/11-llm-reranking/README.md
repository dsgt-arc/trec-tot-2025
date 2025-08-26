# llm reranking

This directory contains supplementary code for wenxin's reranking code.
I'm setting it up so that it's relatively straightforward to run the code.

First make sure you have a copy of https://zenodo.org/records/15356599 locally.
For reranking I only need the queries and qrels, but I've included everything for completeness.

```bash
rclone sync $HOME/scratch/trec-tot-2025/results/rerank gdrive-trec-tot-2025:data/rerank
```

```bash
# about 12 hours
MODEL=google/gemma-3-12b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=gaunernst/gemma-3-12b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=gaunernst/gemma-3-27b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch

# let's see if we can use the quantized models

# about 3 hours 282944
MODEL=gaunernst/gemma-3-1b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
# about 2 hours 283276
MODEL=gaunernst/gemma-3-4b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
# about 8 hours failed in 20 minutes 283277,
MODEL=gaunernst/gemma-3-12b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch

# lets see what happens on the other datasets

# 283351
MODEL=gaunernst/gemma-3-12b-it-qat-compressed-tensors RETRIEVAL_MODEL=bge-passage-dense DEVSET=dev3 sbatch rerank.sbatch

# 10 minutes, 283372, ndcg 32 -> 9, qat is pretty bad
MODEL=gaunernst/gemma-3-12b-it-qat-compressed-tensors RETRIEVAL_MODEL=gemini-2.5-flash DEVSET=dev3 sbatch rerank.sbatch

# what happens when we use the bigger model? 285322 -> 0.0939
MODEL=google/gemma-3-12b-it RETRIEVAL_MODEL=gemini-2.5-flash DEVSET=dev3 sbatch rerank.sbatch

# limit the number of ranked items? 284740, about 1 hour, 0.0929
MODEL=google/gemma-3-12b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch --batch-rank-end 100
# using an unsloth model, 285201, 1 hour, 0.0895 ndcg@10, 20 minute difference
MODEL=unsloth/gemma-3-12b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch --batch-rank-end 100
```
