# LLM Generated Query Performance Results

## PyTerrier BM25 Baseline Results

### LLM-Set1 Performance Evaluation

#### Train Split Results
```bash
trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c $DATA_PATH/2025/generated-queries/llm-set1/train/qrel.txt $TOT/baseline2024/pyterrier-bm25-retrieval/runs/set1-train.txt
```

#### Dev Split Results
```bash
trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c $DATA_PATH/2025/generated-queries/llm-set1/dev/qrel.txt $TOT/baseline2024/pyterrier-bm25-retrieval/runs/set1-dev.txt
```

#### Test Split Results
```bash
trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c $DATA_PATH/2025/generated-queries/llm-set1/test/qrel.txt $TOT/baseline2024/pyterrier-bm25-retrieval/runs/set1-test.txt
```

| Split | MRR    | Recall@1000 | NDCG@10 | NDCG@1000 |
|-------|--------|-------------|---------|-----------|
| Train | 0.0871 | 0.4243      | 0.0963  | 0.1396    |
| Dev   | 0.0964 | 0.4050      | 0.1056  | 0.1436    |
| Test  | 0.0940 | 0.4326      | 0.1034  | 0.1468    |

## For all
| Method | MRR    | Recall@2000 | NDCG@10 | NDCG@1000 |
|--------|--------|-------------|---------|-----------|
| Dense  | 0.0671 | 0.4874      | 0.0756  | 0.1307    |
| Sparse | 0.0891 | 0.4222      | 0.0984  | 0.1409    |
| D+S    | -      | 0.5973      | -       | -         |