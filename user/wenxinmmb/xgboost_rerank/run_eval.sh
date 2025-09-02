# ===========================
# For sparse retrieval
# ===========================

for version in v4 v5 v6; do
  for splitname in train dev1 dev2 dev3 llmset1-train llmset1-dev; do

    python create_features.py \
      --input_file $DATA_PATH/results/$splitname/bm25.txt \
      --input_mode retrieval-dense-tsv \
      --dense_feature_file ${TOT}/baseline2024/dense/outputs/dense_score/all-sets.tsv \
      --sparse_feature_file $DATA_PATH/results/$splitname/bm25.txt \
      --query_file $DATA_PATH/2025/$splitname-2025/queries.jsonl \
      --output_dir outputs/scores/$splitname-bm25/$version

    python evaluate_lamdamart.py \
        --model_path outputs/sample-$version/lambdamart_model.json \
        --baseline_run_path $DATA_PATH/results/$splitname/bm25.txt \
        --dir outputs/scores/$splitname-bm25/$version \
        --qrel_path $DATA_PATH/2025/$splitname-2025/qrel.txt
  done
done

# ===========================
# For dense retrieval
# ===========================
for version in v4 v5 v6; do
  for splitname in train dev1 dev2 dev3 llmset1-train llmset1-dev; do
    echo "Processing split: $splitname, version: $version"

    python create_features.py \
      --input_file $DATA_PATH/results/$splitname/bge-filtered.txt \
      --input_mode retrieval \
      --dense_feature_file $DATA_PATH/results/$splitname/bge-filtered.txt \
      --sparse_feature_file outputs/scores/$splitname-bge/bge-filtered--md-pt.txt \
      --query_file $DATA_PATH/2025/$splitname-2025/queries.jsonl \
      --output_dir outputs/scores/$splitname-bge/$version

    # mkdir -p outputs/scores/$splitname-bge/$version
    # cp outputs/scores/$splitname-bge/v4/features.txt outputs/scores/$splitname-bge/$version
    # cp outputs/scores/$splitname-bge/v4/info.json outputs/scores/$splitname-bge/$version

    python evaluate_lamdamart.py \
        --model_path outputs/sample-$version/lambdamart_model.json \
        --baseline_run_path $DATA_PATH/results/$splitname/bge-filtered.txt \
        --dir outputs/scores/$splitname-bge/$version \
        --qrel_path $DATA_PATH/2025/$splitname-2025/qrel.txt
  done
done