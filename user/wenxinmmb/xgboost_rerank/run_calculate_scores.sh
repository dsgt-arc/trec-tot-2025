#!/bin/bash

# Loop through start-query values
# for start_query in 0 100; do
#   python calculate_scores.py \
#     --mode pt \
#     --retrieval-path $DATA_PATH/results/dev1/bge-filtered.txt \
#     --queries-path $DATA_PATH/2025/dev1-2025/queries.jsonl \
#     --corpus-path $DATA_PATH/2025/corpus.jsonl \
#     --offset-path $DATA_PATH/2025/corpus-offset-mapping.json \
#     --bm25-index-path /home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index \
#     --output-dir outputs/scores/dev1-bge \
#     --start-query $start_query \
#     --num-queries 100 \
#     --no-reorder
# done

# for start_query2 in 100 200 300 400 500; do
#   python calculate_scores.py \
#     --mode pt \
#     --retrieval-path $DATA_PATH/results/dev3/bge-filtered.txt \
#     --queries-path $DATA_PATH/2025/dev3-2025/queries.jsonl \
#     --corpus-path $DATA_PATH/2025/corpus.jsonl \
#     --offset-path $DATA_PATH/2025/corpus-offset-mapping.json \
#     --bm25-index-path /home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index \
#     --output-dir outputs/scores/dev3-bge \
#     --start-query $start_query2 \
#     --num-queries 100 \
#     --no-reorder
# done

# for split in dev1 dev3; do
#   python calculate_scores.py \
#     --mode bge \
#     --retrieval-path $DATA_PATH/results/$split/bm25.txt \
#     --queries-path $DATA_PATH/2025/$split-2025/queries.jsonl \
#     --corpus-path $DATA_PATH/2025/corpus.jsonl \
#     --offset-path $DATA_PATH/2025/corpus-offset-mapping.json \
#     --bm25-index-path /home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index \
#     --output-dir outputs/scores/$split-bm25 \
#     --no-reorder
# done

# split=train
# for split in train dev2; do
#   for start_query1 in 0 100; do
#     python calculate_scores.py \
#       --mode pt \
#       --retrieval-path $DATA_PATH/results/$split/bge-filtered.txt \
#       --queries-path $DATA_PATH/2025/$split-2025/queries.jsonl \
#       --corpus-path $DATA_PATH/2025/corpus.jsonl \
#       --offset-path $DATA_PATH/2025/corpus-offset-mapping.json \
#       --bm25-index-path /home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index \
#       --output-dir outputs/scores/$split-bge \
#       --start-query $start_query1 \
#       --num-queries 100 \
#       --no-reorder
#   done
# done

splitname=llmset1-train

for start_query2 in $(seq 0 100 4100); do
  echo "Processing start query: $start_query2"
  python calculate_scores.py \
    --mode pt \
    --retrieval-path $DATA_PATH/results/$splitname/bge-filtered.txt \
    --queries-path $DATA_PATH/2025/generated-queries/llm-set1/train/queries.jsonl \
    --corpus-path $DATA_PATH/2025/corpus.jsonl \
    --offset-path $DATA_PATH/2025/corpus-offset-mapping.json \
    --bm25-index-path /home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index \
    --output-dir outputs/scores/$splitname-bge \
    --start-query $start_query2 \
    --num-queries 100 \
    --no-reorder
done

# Calculate qrel pt bm25 scores
# for split in train dev1 dev2 dev3 llmset1-train llmset1-dev llmset1-test; do
#   python calculate_scores.py \
#     --mode pt \
#     --retrieval-path $DATA_PATH/2025/$split-2025/qrel.txt \
#     --queries-path $DATA_PATH/2025/$split-2025/queries.jsonl \
#     --corpus-path $DATA_PATH/2025/corpus.jsonl \
#     --offset-path $DATA_PATH/2025/corpus-offset-mapping.json \
#     --bm25-index-path /home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index \
#     --output-dir outputs/scores/qrel-sparse-$split \
#     --no-reorder
# done