# 5 requests: Before 21.53 After 21.44 == 0.1
# start time: 5:24PM 8/30
# end time 12:35 AM 8/31
# end credit: 9.14
python tot_llm_reranking.py     \
    --input-trec-run $DATA_PATH/results/2025-test/comb_500_gemini_bm25_bge.txt \
    --queries-file $DATA_PATH/2025/test-2025/queries.jsonl     \
    --corpus-file $DATA_PATH/2025/corpus.jsonl     \
    --offset-file $DATA_PATH/2025/corpus-offset-mapping.json     \
    --output-dir outputs/2025-test-v1     \
    --use-openrouter     \
    --model google/gemini-2.5-flash     \
    --prompt-template-path custom_templates/rank_lrl_v2.yaml     \
    --document-mode title_only     \
    --batch-window-size 105     \
    --batch-stride 100     \
    --batch-rank-end 1100  \
    --save-invocations-history     \
    --start-query-index 517 \
    --model-context-size 1000000