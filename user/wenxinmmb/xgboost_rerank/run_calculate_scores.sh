#!/bin/bash

# ==================================
# Calculate BGE dense scores for sparse retrieval results
# ==================================

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

# ==================================
# Calculate pt bm25 scores for dense retrieval results
# ==================================
# for split in dev2; do
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

split=test
for start_query1 in 0 100 200 300 400 500 600; do
  python calculate_scores.py \
    --mode pt \
    --retrieval-path $DATA_PATH/results/2025-$split/llm-gemini.txt \
    --queries-path $DATA_PATH/2025/$split-2025/queries.jsonl \
    --corpus-path $DATA_PATH/2025/corpus.jsonl \
    --offset-path $DATA_PATH/2025/corpus-offset-mapping.json \
    --bm25-index-path /home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index \
    --output-dir outputs/scores/2025-$split-gemini \
    --start-query $start_query1 \
    --num-queries 100 \
    --no-reorder
done

# ==================================
# Calculate pt bm25 scores for QREL files
# ==================================
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