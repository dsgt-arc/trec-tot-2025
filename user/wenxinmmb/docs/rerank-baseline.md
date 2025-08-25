# Baseline Retrieval Results Comparison

## Overview
Evaluation results for 100 queries across two datasets using BM25, BGE, and LLM retrievers.

## Results Summary

### Dev3-2025 Dataset (First 100 queries)

| Method | Recall@10 | NDCG@10 | Recall@100 | NDCG@100 | Recall@1000 | NDCG@1000 | Recall@2000 | NDCG@2000 |
|--------|-----------|---------|-------------|----------|-------------|-----------|-------------|-----------|
| **LLM** | **0.7000** | **0.6609** | **0.7000** | **0.6609** | **0.7000** | **0.6609** | **0.7000** | **0.6609** |
| BM25 | 0.4400 | 0.3302 | 0.6200 | 0.3665 | 0.8100 | 0.3894 | 0.8100 | 0.3894 |
| BGE | 0.2300 | 0.1667 | 0.3900 | 0.2007 | 0.6400 | 0.2315 | 0.6400 | 0.2315 |

### LLM-Set1 Train Dataset (100 queries)

| Method | Recall@10 | NDCG@10 | Recall@100 | NDCG@100 | Recall@1000 | NDCG@1000 | Recall@2000 | NDCG@2000 |
|--------|-----------|---------|-------------|----------|-------------|-----------|-------------|-----------|
| **LLM** | **0.3500** | **0.2899** | **0.3500** | **0.2899** | **0.3500** | **0.2899** | **0.3500** | **0.2899** |
| BM25 | 0.1200 | 0.0896 | 0.3100 | 0.1285 | 0.4800 | 0.1499 | 0.4800 | 0.1499 |
| BGE | 0.1100 | 0.0714 | 0.2100 | 0.0911 | 0.4500 | 0.1199 | 0.4500 | 0.1199 |

## Key Observations

### Performance by Dataset
- **Dev3-2025**: LLM method achieves the highest precision metrics
  - LLM: 70% recall@10 with 0.66 NDCG@10 (consistent across all cutoffs)
  - BM25: 44% recall@10, improving to 81% recall@1000
  - BGE: 23% recall@10, improving to 64% recall@1000
- **LLM-Set1**: LLM shows strong performance at early cutoffs
  - LLM: 35% recall@10 (constant across all cutoffs)
  - BM25: 12% recall@10, improving to 48% recall@1000
  - BGE: 11% recall@10, improving to 45% recall@1000

### Method Comparison
- **LLM method shows unique characteristics**:
  - **Consistent performance across all cutoff levels** (no improvement beyond @10)
  - **Highest precision**: Best NDCG@10 on both datasets
  - **Limited recall growth**: Performance plateaus immediately
- **BM25 vs BGE**: BM25 consistently outperforms BGE, especially at higher cutoffs
- **Trade-offs**: LLM excels at precision, BM25 at recall

### Dataset Characteristics
- Dev3-2025 appears to be an easier retrieval task overall
- LLM-Set1 shows more challenging retrieval characteristics
- **LLM method performance gap** between datasets (70% vs 35% recall) suggests dataset sensitivity

## Combined Results (All Methods)

### Dev3-2025 Dataset - Combined Retrieval Methods

| Method | Recall@10 | NDCG@10 | Recall@100 | NDCG@100 | Recall@1000 | NDCG@1000 | Recall@2000 | NDCG@2000 | Recall@3000 | NDCG@3000 |
|--------|-----------|---------|-------------|----------|-------------|-----------|-------------|-----------|-------------|-----------|
| **Combined (500 each)** | **0.7000** | **0.6609** | **0.7700** | **0.6748** | **0.8800** | **0.6878** | **0.8800** | **0.6878** | **0.8800** | **0.6878** |
| **Combined (All)** | **0.7300** | **0.6662** | **0.7500** | **0.6705** | **0.8800** | **0.6865** | **0.9000** | **0.6884** | **0.9000** | **0.6884** |

### LLM-Set1 Train Dataset - Combined Retrieval Methods

| Method | Recall@10 | NDCG@10 | Recall@100 | NDCG@100 | Recall@1000 | NDCG@1000 | Recall@2000 | NDCG@2000 | Recall@3000 | NDCG@3000 |
|--------|-----------|---------|-------------|----------|-------------|-----------|-------------|-----------|-------------|-----------|
| **Combined (500 each)** | **0.3500** | **0.2899** | **0.4900** | **0.3178** | **0.6600** | **0.3387** | **0.6600** | **0.3387** | **0.6600** | **0.3387** |
| **Combined (All)** | **0.3800** | **0.2858** | **0.5000** | **0.3101** | **0.6600** | **0.3294** | **0.7000** | **0.3332** | **0.7000** | **0.3332** |

### Combined Method Analysis
- **Combination order**: LLM → BM25 → BGE
- **"Combined (500 each)"**: Takes top 500 results from each method
- **"Combined (All)"**: Uses all available results from each method

#### Key Insights:
- **Combined methods show improved recall** at higher cutoffs compared to individual methods
- **Dev3-2025**: Combined methods achieve 88-90% recall@1000+ vs individual method peaks of 81% (BM25)
- **LLM-Set1**: Combined methods reach 66-70% recall@1000+ vs individual method peaks of 48% (BM25)
- **Taking all results vs 500 each**:
  - Dev3-2025: "All" slightly better at @10 and @2000+
  - LLM-Set1: "All" performs better across most metrics
- **Best overall performance**: Combined methods maintain competitive NDCG while significantly improving recall

---

## LLM rerank results
### Dev3-2025 Dataset (First 100 queries)

Rerank configuration: `rerank_module/outputs/dev3-100-gemma12b-fs-v13/` and `rerank_module/outputs/llmset1-t100v1-gemini-fs-v1`

```
{
  "model": "google/gemini-2.5-flash",
  "api_base": "https://openrouter.ai/api/v1",
  "document_mode": "title_only",
  "batch_window_size": 105,
  "batch_stride": 100,
  "template_path": "custom_templates/rank_lrl_v2.yaml"
}
```

### Dev3-2025 Dataset (First 100 queries)
| Method | Recall@10 | NDCG@10 | Recall@100 | NDCG@100 | Recall@1000 | NDCG@1000 | Recall@2000 | NDCG@2000 |
|--------|-----------|---------|-------------|----------|-------------|-----------|-------------|-----------|
| BM25 (before) | 0.4400 | 0.3302 | 0.6200 | 0.3665 | 0.8100 | 0.3894 | 0.8100 | 0.3894 |
| BM25 (after) | 0.7200 | 0.6798 | 0.7400 | 0.6849 | 0.8100 | 0.6937 | 0.8100 | 0.6937 |

### LLM-Set1 Train Dataset (100 queries)
| Method | Recall@10 | NDCG@10 | Recall@100 | NDCG@100 | Recall@1000 | NDCG@1000 | Recall@2000 | NDCG@2000 |
|--------|-----------|---------|-------------|----------|-------------|-----------|-------------|-----------|
| BM25 (before) | 0.1200 | 0.0896 | 0.3100 | 0.1285 | 0.4800 | 0.1499 | 0.4800 | 0.1499 |
| BM25 (after) | 0.4400 | 0.3763 | 0.4500 | 0.3779 | 0.4800 | 0.3816 | 0.4800 | 0.3816 |
| BM25 (after gemini flash-lite) | 0.4000 | 0.2819 | 0.4400 | 0.2881 | 0.4800 | 0.2938 | 0.4800 | 0.2938 |

BM25 (after gemini flash-lite): is not competitive; its directory is `rerank_module/outputs/llmset1-t100v1-gemini-fs-v2`; in short; first-sentence, gemini-2.5-flash-lite


## Command Reference
```bash
# Dev3-2025 evaluations
python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/dev3-2025/qrel-first-100.txt $DATA_PATH/results/dev3-100/bge.txt
python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/dev3-2025/qrel-first-100.txt $DATA_PATH/results/dev3-100/bm25.txt
python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/dev3-2025/qrel-first-100.txt $DATA_PATH/results/dev3-100/llm.txt

# Dev3-2025 combined evaluations
python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/dev3-2025/qrel-first-100.txt $DATA_PATH/results/dev3-100/comb_500.txt
python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/dev3-2025/qrel-first-100.txt $DATA_PATH/results/dev3-100/all.txt

# Dev3-2025 LLM reranked evaluations
$ python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/dev3-2025/qrel-first-100.txt $TOT/rerank_module/outputs/dev3-100-gemma12b-fs-v13/rerank-results-reformatted.txt

# LLM-Set1 evaluations  
python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/generated-queries/llm-set1/train/train-100-v1/qrel.txt $DATA_PATH/results/llmset1-train-100-v1/bge.txt
python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/generated-queries/llm-set1/train/train-100-v1/qrel.txt $DATA_PATH/results/llmset1-train-100-v1/bm25.txt
python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/generated-queries/llm-set1/train/train-100-v1/qrel.txt $DATA_PATH/results/llmset1-train-100-v1/llm.txt

# LLM-Set1 combined evaluations
python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/generated-queries/llm-set1/train/train-100-v1/qrel.txt $DATA_PATH/results/llmset1-train-100-v1/comb_500.txt
python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/generated-queries/llm-set1/train/train-100-v1/qrel.txt $DATA_PATH/results/llmset1-train-100-v1/all.txt

# LLM-Set1 LLM reranked evaluations
$ python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/generated-queries/llm-set1/train/train-100-v1/qrel.txt $TOT/rerank_module/outputs/llmset1-t100v1-gemini-fs-v1/rerank-results-reformatted.txt
$ python $TOT/fastrank_imp/manual_scores.py $DATA_PATH/2025/generated-queries/llm-set1/train/train-100-v1/qrel.txt $TOT/rerank_module/outputs/llmset1-t100v1-gemini-fs-v2/rerank-results-reformatted.txt

```
