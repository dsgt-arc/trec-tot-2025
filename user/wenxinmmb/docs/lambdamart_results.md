# LambdaMART Model for Reranking

This document presents the results of training a LambdaMART model for reranking documents in the TREC ToT 2025 task.

## |     1000 |            0.13 | 0.2000 (47.54%) | 0.1900 (43.05%) | 0.1800 (38.23%) |              0.48 | 0.4800 (0.00%)  | 0.4800 (0.00%)  | 0.4800 (0.00%)  |

## Results Analysis

### Model Performance Comparison

**Overall Best Model: v6** consistently shows the strongest performance across training datasets, with the highest improvements in most scenarios.

**Performance by Model Version:**

- **v4**: Solid baseline improvements but generally the most conservative gains
- **v5**: Moderate improvements, often falling between v4 and v6
- **v6**: Consistently highest performance gains, especially on training sets

### Overfitting Analysis

**Evidence of Overfitting:**

1. **Training vs. Validation Performance Gap**: 
   - Training sets show dramatic improvements (e.g., train|bm25: 94% improvement at @10 for v6)
   - Validation sets (dev1, dev2) show much smaller gains or even negative performance
   - This large gap indicates the model is memorizing training patterns rather than generalizing

2. **Performance Degradation on Unseen Data**:
   - Dev1|BM25: v6 shows negative performance (-14% at @10)
   - Dev2|BM25: v6 shows declining performance compared to v4 and v5
   - LLMSet1-dev shows consistent degradation from v4 → v5 → v6

3. **Hyperparameter Complexity**:
   - v6 uses more complex parameters (max_depth=8, gamma=0.5) which can lead to overfitting
   - v5 with simpler, more regularized parameters shows better generalization

### Training Set Expectations

**Expected High Performance on Training Sets:**
- Models are trained on portions of these datasets (train, dev3 first 200, llmset1-train)
- High improvements are expected and normal (50-175% improvements observed)
- These results validate that the model is learning the training signal effectively

**Concerning Patterns:**
- The extremely high improvements (>100%) suggest potential overfitting
- Real-world performance should be evaluated on completely held-out test sets

### Recommendations

1. **Use v5 for Production**: Shows better balance between training performance and generalization
2. **Implement Regularization**: Add early stopping, reduce max_depth, increase min_child_weight
3. **Cross-Validation**: Use proper k-fold validation to get more reliable performance estimates
4. **Feature Selection**: Consider feature importance analysis to reduce overfitting
5. **More Training Data**: Increase training set size to improve generalization

### Key Insights

- **Dense vs. Sparse**: BGE dense retrieval generally shows larger improvements than BM25
- **Cutoff Effects**: Improvements are most pronounced at lower cutoffs (@10, @100)
- **Dataset Dependency**: Performance varies significantly across different query sets
- **Hyperparameter Sensitivity**: Small parameter changes lead to significant performance differencesverview

This project trains a LambdaMART model for reranking retrieved documents. The model leverages multiple features including dense retrieval scores, sparse retrieval scores, query characteristics, and document popularity metrics to improve ranking performance.

## Sampling Strategy

The training set is constructed using queries from:
- `train` dataset
- `dev3` dataset (first 200 queries)
- `llmset1-train` dataset

For each query, documents are sampled with the following relevance assignments:
- **Correct document**: Relevance score of 2
- **Pseudo-relevant documents**: 5 documents selected from top-10 results of both dense and sparse retrieval (relevance score of 1)
- **Irrelevant documents**: 10 documents selected from positions 11-1000 in retrieval results (relevance score of 0)

## Features

The model uses 5 features for ranking:

1. **Dense score** from `all-sets.tsv` (raw score)
2. **Sparse score** from `bm25.txt` (min-max normalized in range [0-100])
3. **Query word count** (min-max normalized)
4. **Pageview count** (log1p + z-score normalized)
5. **PageRank score** (log + z-score normalized)

## Training and Hyperparameter Tuning

### v4 - Initial Parameters
```python
params = {
    "objective": "rank:pairwise",  # Pairwise ranking objective
    "eval_metric": "ndcg",         # Evaluation metric
    "eta": 0.1,                    # Learning rate
    "max_depth": 6,                # Maximum tree depth
    "min_child_weight": 1,         # Minimum sum of instance weight (hessian) needed in a child
    "gamma": 0.0,                  # Minimum loss reduction required to make a further partition
    "subsample": 0.8,              # Subsample ratio of the training instances
    "colsample_bytree": 0.8        # Subsample ratio of columns when constructing each tree
}
```

### v5 - Grid Search Results
Best parameters from grid search:
```python
best_params = {
    "colsample_bytree": 1.0,
    "eta": 0.2,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 1.0
}
```

### v6 - Random Search Results
Best parameters from random search:
```python
best_params = {
    "eta": 0.2,
    "max_depth": 8,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.5
}
```

## Results

The following tables show the performance comparison between baseline and the three model versions (v4, v5, v6) across different datasets and retrieval methods. Percentages in parentheses indicate relative improvement over baseline.

## Training Set Results - BM25 Retrieval

|   Cutoff |   Baseline NDCG | v4 NDCG         | v5 NDCG         | v6 NDCG         |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall        |
|---------:|----------------:|:----------------|:----------------|:----------------|------------------:|:----------------|:----------------|:-----------------|
|       10 |            0.06 | 0.1000 (59.11%) | 0.1100 (62.18%) | 0.1300 (94.23%) |              0.1  | 0.1500 (46.67%) | 0.1500 (40.00%) | 0.2100 (100.00%) |
|      100 |            0.09 | 0.1300 (48.05%) | 0.1400 (51.25%) | 0.1600 (74.35%) |              0.24 | 0.3000 (26.47%) | 0.2900 (23.53%) | 0.3600 (50.00%)  |
|     1000 |            0.12 | 0.1500 (32.26%) | 0.1600 (35.61%) | 0.1700 (46.77%) |              0.45 | 0.4500 (0.00%)  | 0.4500 (0.00%)  | 0.4500 (0.00%)   |

## Training Set Results - BGE Dense Retrieval

|   Cutoff |   Baseline NDCG | v4 NDCG          | v5 NDCG          | v6 NDCG          |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall        |
|---------:|----------------:|:-----------------|:-----------------|:-----------------|------------------:|:----------------|:----------------|:-----------------|
|       10 |            0.04 | 0.0900 (118.80%) | 0.0900 (115.47%) | 0.1200 (175.48%) |              0.08 | 0.1300 (50.00%) | 0.1200 (41.67%) | 0.1900 (125.00%) |
|      100 |            0.06 | 0.1200 (88.62%)  | 0.1200 (95.61%)  | 0.1500 (142.17%) |              0.2  | 0.2700 (35.71%) | 0.2900 (46.43%) | 0.3800 (96.43%)  |
|     1000 |            0.1  | 0.1500 (51.05%)  | 0.1500 (53.15%)  | 0.1700 (70.58%)  |              0.47 | 0.4700 (0.00%)  | 0.4700 (0.00%)  | 0.4700 (0.00%)   |

## Dev1 Results - BM25 Retrieval
|   Cutoff |   Baseline NDCG | v4 NDCG         | v5 NDCG        | v6 NDCG          |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall       |
|---------:|----------------:|:----------------|:---------------|:-----------------|------------------:|:----------------|:----------------|:----------------|
|       10 |            0.08 | 0.0900 (11.04%) | 0.0900 (7.47%) | 0.0700 (-14.11%) |              0.11 | 0.1400 (33.33%) | 0.1400 (33.33%) | 0.1100 (6.67%)  |
|      100 |            0.11 | 0.1100 (2.88%)  | 0.1100 (4.62%) | 0.1000 (-3.79%)  |              0.22 | 0.2300 (3.23%)  | 0.2500 (16.13%) | 0.2500 (16.13%) |
|     1000 |            0.13 | 0.1400 (2.48%)  | 0.1400 (1.30%) | 0.1300 (-5.09%)  |              0.45 | 0.4500 (0.00%)  | 0.4500 (0.00%)  | 0.4500 (0.00%)  |

## Dev1 Results - BGE Dense Retrieval
|   Cutoff |   Baseline NDCG | v4 NDCG          | v5 NDCG         | v6 NDCG         |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall       |
|---------:|----------------:|:-----------------|:----------------|:----------------|------------------:|:----------------|:----------------|:----------------|
|       10 |            0.05 | 0.1000 (103.20%) | 0.1000 (97.28%) | 0.0900 (76.17%) |              0.1  | 0.1500 (57.14%) | 0.1400 (42.86%) | 0.1300 (35.71%) |
|      100 |            0.07 | 0.1300 (74.56%)  | 0.1200 (66.07%) | 0.1200 (62.01%) |              0.23 | 0.3000 (34.37%) | 0.2700 (18.75%) | 0.3000 (34.37%) |
|     1000 |            0.11 | 0.1500 (44.16%)  | 0.1500 (43.16%) | 0.1400 (36.04%) |              0.48 | 0.4800 (0.00%)  | 0.4800 (0.00%)  | 0.4800 (0.00%)  |

## Dev2 Results - BM25 Retrieval
|   Cutoff |   Baseline NDCG | v4 NDCG         | v5 NDCG         | v6 NDCG          |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall       |
|---------:|----------------:|:----------------|:----------------|:-----------------|------------------:|:----------------|:----------------|:----------------|
|       10 |            0.1  | 0.1100 (13.03%) | 0.1000 (-2.99%) | 0.0900 (-11.79%) |              0.15 | 0.1700 (14.29%) | 0.1500 (4.76%)  | 0.1300 (-9.52%) |
|      100 |            0.12 | 0.1400 (16.13%) | 0.1200 (1.44%)  | 0.1200 (0.74%)   |              0.23 | 0.2900 (24.24%) | 0.2600 (12.12%) | 0.2800 (21.21%) |
|     1000 |            0.14 | 0.1600 (9.14%)  | 0.1400 (-0.07%) | 0.1400 (-2.72%)  |              0.45 | 0.4500 (0.00%)  | 0.4500 (0.00%)  | 0.4500 (0.00%)  |

## Dev2 Results - BGE Dense Retrieval
|   Cutoff |   Baseline NDCG | v4 NDCG         | v5 NDCG         | v6 NDCG         |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall       |
|---------:|----------------:|:----------------|:----------------|:----------------|------------------:|:----------------|:----------------|:----------------|
|       10 |            0.07 | 0.1000 (54.43%) | 0.0900 (38.06%) | 0.0800 (25.35%) |              0.11 | 0.1700 (50.00%) | 0.1600 (43.75%) | 0.1500 (37.50%) |
|      100 |            0.1  | 0.1200 (27.23%) | 0.1100 (16.07%) | 0.1100 (12.18%) |              0.27 | 0.2800 (2.56%)  | 0.2700 (0.00%)  | 0.2900 (7.69%)  |
|     1000 |            0.13 | 0.1500 (20.55%) | 0.1400 (12.86%) | 0.1400 (7.95%)  |              0.51 | 0.5100 (0.00%)  | 0.5100 (0.00%)  | 0.5100 (0.00%)  |

## Dev3 Results - BM25 Retrieval
|   Cutoff |   Baseline NDCG | v4 NDCG         | v5 NDCG         | v6 NDCG         |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall       |
|---------:|----------------:|:----------------|:----------------|:----------------|------------------:|:----------------|:----------------|:----------------|
|       10 |            0.34 | 0.4100 (22.54%) | 0.3900 (15.13%) | 0.4100 (20.43%) |              0.43 | 0.5300 (21.46%) | 0.4900 (12.02%) | 0.5100 (18.45%) |
|      100 |            0.37 | 0.4400 (19.09%) | 0.4300 (14.76%) | 0.4300 (17.13%) |              0.6  | 0.6700 (11.18%) | 0.6600 (10.56%) | 0.6600 (9.32%)  |
|     1000 |            0.39 | 0.4600 (16.18%) | 0.4400 (12.15%) | 0.4500 (14.67%) |              0.77 | 0.7700 (0.00%)  | 0.7700 (0.00%)  | 0.7700 (0.00%)  |

## Dev3 Results - BGE Dense Retrieval
|   Cutoff |   Baseline NDCG | v4 NDCG          | v5 NDCG          | v6 NDCG          |   Baseline Recall | v4 Recall        | v5 Recall        | v6 Recall        |
|---------:|----------------:|:-----------------|:-----------------|:-----------------|------------------:|:-----------------|:-----------------|:-----------------|
|       10 |            0.14 | 0.3700 (173.05%) | 0.3500 (155.19%) | 0.3600 (161.96%) |              0.21 | 0.4600 (119.47%) | 0.4300 (105.31%) | 0.4500 (111.50%) |
|      100 |            0.17 | 0.4000 (128.46%) | 0.3800 (116.60%) | 0.3900 (121.31%) |              0.4  | 0.5900 (47.44%)  | 0.5800 (43.72%)  | 0.5800 (45.58%)  |
|     1000 |            0.2  | 0.4000 (97.64%)  | 0.3900 (88.57%)  | 0.3900 (92.19%)  |              0.65 | 0.6500 (0.00%)   | 0.6500 (0.00%)   | 0.6500 (0.00%)   |

## LLMSet1-Train Results - BM25 Retrieval
|   Cutoff |   Baseline NDCG | v4 NDCG         | v5 NDCG         | v6 NDCG         |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall       |
|---------:|----------------:|:----------------|:----------------|:----------------|------------------:|:----------------|:----------------|:----------------|
|       10 |            0.1  | 0.1600 (70.21%) | 0.1600 (70.01%) | 0.1800 (88.34%) |              0.14 | 0.2300 (60.21%) | 0.2300 (61.97%) | 0.2500 (77.11%) |
|      100 |            0.12 | 0.1900 (60.85%) | 0.1900 (60.93%) | 0.2100 (74.59%) |              0.25 | 0.3600 (42.56%) | 0.3700 (44.04%) | 0.3800 (49.46%) |
|     1000 |            0.14 | 0.2000 (42.94%) | 0.2000 (42.64%) | 0.2100 (53.02%) |              0.42 | 0.4200 (0.00%)  | 0.4200 (0.00%)  | 0.4200 (0.00%)  |

## LLMSet1-Train Results - BGE Dense Retrieval
|   Cutoff |   Baseline NDCG | v4 NDCG          | v5 NDCG          | v6 NDCG          |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall        |
|---------:|----------------:|:-----------------|:-----------------|:-----------------|------------------:|:----------------|:----------------|:-----------------|
|       10 |            0.08 | 0.1700 (109.94%) | 0.1700 (113.20%) | 0.1900 (140.79%) |              0.13 | 0.2400 (85.80%) | 0.2400 (90.14%) | 0.2700 (110.85%) |
|      100 |            0.11 | 0.2000 (84.26%)  | 0.2000 (86.60%)  | 0.2200 (104.81%) |              0.27 | 0.4000 (46.51%) | 0.4000 (48.62%) | 0.4200 (52.75%)  |
|     1000 |            0.13 | 0.2100 (57.06%)  | 0.2100 (58.36%)  | 0.2300 (72.05%)  |              0.49 | 0.4900 (0.00%)  | 0.4900 (0.00%)  | 0.4900 (0.00%)   |

## LLMSet1-Dev Results - BM25 Retrieval
|   Cutoff |   Baseline NDCG | v4 NDCG         | v5 NDCG         | v6 NDCG         |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall       |
|---------:|----------------:|:----------------|:----------------|:----------------|------------------:|:----------------|:----------------|:----------------|
|       10 |            0.11 | 0.1500 (44.50%) | 0.1400 (36.09%) | 0.1400 (27.90%) |              0.15 | 0.2100 (44.92%) | 0.2100 (44.07%) | 0.2000 (36.44%) |
|      100 |            0.12 | 0.1800 (44.79%) | 0.1700 (38.61%) | 0.1600 (33.82%) |              0.24 | 0.3400 (44.68%) | 0.3400 (45.74%) | 0.3500 (47.34%) |
|     1000 |            0.14 | 0.1900 (30.04%) | 0.1800 (24.55%) | 0.1700 (20.03%) |              0.4  | 0.4000 (0.00%)  | 0.4000 (0.00%)  | 0.4000 (0.00%)  |

## LLMSet1-Dev Results - BGE Dense Retrieval
|   Cutoff |   Baseline NDCG | v4 NDCG         | v5 NDCG         | v6 NDCG         |   Baseline Recall | v4 Recall       | v5 Recall       | v6 Recall       |
|---------:|----------------:|:----------------|:----------------|:----------------|------------------:|:----------------|:----------------|:----------------|
|       10 |            0.08 | 0.1500 (93.62%) | 0.1400 (83.99%) | 0.1300 (75.44%) |              0.12 | 0.2100 (77.89%) | 0.2000 (72.63%) | 0.2000 (69.47%) |
|      100 |            0.11 | 0.1800 (69.38%) | 0.1800 (64.17%) | 0.1700 (57.37%) |              0.27 | 0.3800 (38.36%) | 0.3800 (39.27%) | 0.3700 (36.53%) |
|     1000 |            0.13 | 0.2000 (47.54%) | 0.1900 (43.05%) | 0.1800 (38.23%) |              0.48 | 0.4800 (0.00%)  | 0.4800 (0.00%)  | 0.4800 (0.00%)  