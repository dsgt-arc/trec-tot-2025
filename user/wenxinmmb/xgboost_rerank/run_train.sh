# python create_features.py --input_file outputs/sample-v4/sample.txt \
#     --input_mode sample-precomputed \
#     --dense_feature_file none \
#     --sparse_feature_file none \
#     --query_files $DATA_PATH/2025/train-2025/queries.jsonl $DATA_PATH/2025/dev3-2025/queries.jsonl $DATA_PATH/2025/llmset1-train-2025/queries.jsonl \
#     --output_dir outputs/sample-v4

python train_lambdamart.py \
    --dir outputs/sample-v4 \

