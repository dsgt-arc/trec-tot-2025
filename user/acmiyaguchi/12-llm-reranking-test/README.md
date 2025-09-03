# llm reranking for test

```
comb_200_gemini_bm25_bge
lambdamart-v5-all-1000


export RETRIEVAL_MODEL=comb_200_gemini_bm25_bge
export START_QUERY_INDEX=0
sbatch -J "$RETRIEVAL_MODEL-$START_QUERY_INDEX" rerank.sbatch
```
