# ===========================
# For sparse retrieval
# ===========================
for version in v4 v5 v6; do
  for splitname in test; do
    echo "================================================="
    echo "Processing Sparse retrieval split: $splitname, version: $version"

    python create_features.py \
      --input_file $DATA_PATH/results/2025-$splitname/pt-bm25.txt \
      --input_mode retrieval-dense-tsv \
      --dense_feature_file ${TOT}/baseline2024/dense/outputs/dense_score/2025-test-comb-all.tsv \
      --sparse_feature_file $DATA_PATH/results/2025-$splitname/pt-bm25.txt \
      --query_files $DATA_PATH/2025/$splitname-2025/queries.jsonl \
      --output_dir outputs/scores/2025-$splitname-bm25/$version

    python evaluate_lamdamart.py \
        --model_path outputs/sample-$version/lambdamart_model.json \
        --dir outputs/scores/2025-$splitname-bm25/$version \
        --skip_eval
  done
done

# ===========================
# For LLM retrieval
# ===========================
for version in v4 v5 v6; do
  for splitname in test; do
    echo "================================================="
    echo "Processing LLM retrieval split: $splitname, version: $version"

    python create_features.py \
      --input_file $DATA_PATH/results/2025-$splitname/llm-gemini.txt \
      --input_mode retrieval-dense-tsv \
      --dense_feature_file ${TOT}/baseline2024/dense/outputs/dense_score/2025-test-comb-all.tsv \
      --sparse_feature_file outputs/scores/2025-$splitname-gemini/llm-gemini--md-pt.txt \
      --query_files $DATA_PATH/2025/$splitname-2025/queries.jsonl \
      --output_dir outputs/scores/2025-$splitname-gemini/$version

    python evaluate_lamdamart.py \
        --model_path outputs/sample-$version/lambdamart_model.json \
        --dir outputs/scores/2025-$splitname-gemini/$version \
        --skip_eval
  done
done


# ===========================
# For dense retrieval
# ===========================
for version in v4 v5 v6; do
  for splitname in test; do
    echo "================================================="
    echo "Processing split: $splitname, version: $version"

    python create_features.py \
      --input_file $DATA_PATH/results/2025-$splitname/bge-filtered.txt \
      --input_mode retrieval \
      --dense_feature_file $DATA_PATH/results/2025-$splitname/bge-filtered.txt \
      --sparse_feature_file outputs/scores/2025-$splitname-bge/bge-filtered--md-pt.txt \
      --query_files $DATA_PATH/2025/$splitname-2025/queries.jsonl \
      --output_dir outputs/scores/2025-$splitname-bge/$version

    python evaluate_lamdamart.py \
        --model_path outputs/sample-$version/lambdamart_model.json \
        --dir outputs/scores/2025-$splitname-bge/$version \
        --skip_eval
  done
done