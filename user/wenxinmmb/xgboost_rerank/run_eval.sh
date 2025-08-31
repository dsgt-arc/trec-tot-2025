# For sparse retrieval
# python create_features.py \
#   --input_file $DATA_PATH/results/dev1/bm25.txt \
#   --input_mode retrieval \
#   --dense_feature_file outputs/scores/dev1-bm25/bm25--md-bge-dense.txt \
#   --sparse_feature_file $DATA_PATH/results/dev1/bm25.txt \
#   --query_file $DATA_PATH/2025/dev1-2025/queries.jsonl \
#   --output_dir outputs/scores/dev1-bm25/v2


# python evaluate_lamdamart.py \
#     --model_path outputs/sample-v1/feat-v2/lambdamart_model.json \
#     --baseline_run_path $DATA_PATH/results/dev1/bm25.txt \
#     --dir outputs/scores/dev1-bm25/v2 \
#     --qrel_path $DATA_PATH/2025/dev1-2025/qrel.txt

# For dense retrieval
# splitname=dev3
splitname=llmset1-dev
python create_features.py \
  --input_file $DATA_PATH/results/$splitname/bge-filtered.txt \
  --input_mode retrieval \
  --dense_feature_file $DATA_PATH/results/$splitname/bge-filtered.txt \
  --sparse_feature_file outputs/scores/$splitname-bge/bge-filtered--md-pt.txt \
  --query_file $DATA_PATH/2025/$splitname-2025/queries.jsonl \
  --output_dir outputs/scores/$splitname-bge/v2


python evaluate_lamdamart.py \
    --model_path outputs/sample-v1/feat-v2/lambdamart_model.json \
    --baseline_run_path $DATA_PATH/results/$splitname/bge-filtered.txt \
    --dir outputs/scores/$splitname-bge/v2 \
    --qrel_path $DATA_PATH/2025/$splitname-2025/qrel.txt
