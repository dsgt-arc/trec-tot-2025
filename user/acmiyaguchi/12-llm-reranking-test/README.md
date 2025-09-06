# llm reranking for test

```bash
comb_200_gemini_bm25_bge
lambdamart-v5-all-1000

export RETRIEVAL_MODEL=comb_200_gemini_bm25_bge
export START_QUERY_INDEX=0
sbatch -J "$RETRIEVAL_MODEL-$START_QUERY_INDEX" rerank.sbatch
```

There are 622 queries in the test dataset:

```bash
$ cat ~/trec-tot-2025/.data/official/test-2025-queries.jsonl | wc -l
622
```

Let's do it for a bunch of these:

```bash
export RETRIEVAL_MODEL=comb_200_gemini_bm25_bge
for idx in $(seq 0 50 600); do
   echo \
   RETRIEVAL_MODEL=$RETRIEVAL_MODEL \
   START_QUERY_INDEX=$idx \
   sbatch -J "$RETRIEVAL_MODEL-$idx" rerank.sbatch
done
```

```bash
export RETRIEVAL_MODEL=comb_200_gemini_bm25_bge
for idx in 150 300 400 450; do
   echo \
   RETRIEVAL_MODEL=$RETRIEVAL_MODEL \
   START_QUERY_INDEX=$idx \
   sbatch -J "$RETRIEVAL_MODEL-$idx" rerank.sbatch
done
```

Now concatenate the results:

```bash
path=$HOME/trec-tot-2025/.scratch/results/rerank/v7/comb_200_gemini_bm25_bge
# count the number of results per id
cat $path/*/*results-reformatted.txt | cut -d' ' -f1 | sort  | uniq -c | wc -l
# turns out there are 520 results, which ones am I missing?

cat $HOME/trec-tot-2025/.scratch/data/official/test-2025-queries.jsonl | jq -r '.query_id' | sort

# Create a comparison to find missing queries
path=$HOME/trec-tot-2025/.scratch/results/rerank/v7/comb_200_gemini_bm25_bge
cat $path/*/*results-reformatted.txt | cut -d' ' -f1 | sort | uniq > /tmp/processed_queries.txt
cat $HOME/trec-tot-2025/.scratch/data/official/test-2025-queries.jsonl | jq -r '.query_id' | sort > /tmp/all_queries.txt

# Show the difference (missing queries)
echo "Missing queries:"
comm -23 /tmp/all_queries.txt /tmp/processed_queries.txt

# Get summary counts
echo "Total queries: $(wc -l < /tmp/all_queries.txt)"
echo "Processed queries: $(wc -l < /tmp/processed_queries.txt)"
echo "Missing queries: $(comm -23 /tmp/all_queries.txt /tmp/processed_queries.txt | wc -l)"

# Run the analysis script to identify which batches to rerun
python find_missing_batches.py
```

Now I'm missing two documents, but I'm going to submit this as is for now.

```bash
inpath=$HOME/trec-tot-2025/.scratch/results/rerank/v7/comb_200_gemini_bm25_bge
outpath=$HOME/scratch/trec-tot-2025/data/rerank_test/v7_comb_200_gemini_bm25_bge/combined
mkdir -p $outpath
cat $inpath/*/*results-reformatted.txt > $outpath/results.txt
cp -r $inpath $(dirname $outpath)/raw

# sync
rclone copy $HOME/scratch/trec-tot-2025/data/rerank_test gdrive-trec-tot-2025:data/rerank_test
```
