# Overview
- Reproduce LLM query generation from https://github.com/kimdanny/llm-tot-query-elicitation  
- Setup: test first 100 queries in dev3
- Models: compare queries generated from gemini-2.5-flash-lite and gpt-o4-mini with the orignal trec queries

# Results
## 1. Correlation coeffients of query embeddings
Embedding model: all-MiniLM-L6-v2
```
set 1: original TREC query
set 2: query from gemini-flash-lite
set 3: query from gpt-4o-mini
set 4: query from first 200 words in Wikipedia page  
set 5: random text 200 words

Pearson correlation set 1 and set 2: 0.9372  
Pearson P-value: 0.0000  
Kendall's Tau correlation set 1 and set 2: 0.7771  
Kendall's Tau P-value: 0.0000  

Pearson correlation set 1 and set 3: 0.9375  
Pearson P-value: 0.0000  
Kendall's Tau correlation set 1 and set 3: 0.7772  
Kendall's Tau P-value: 0.0000  

Pearson correlation set 1 and set 4: 0.6199  
Pearson P-value: 0.0000  
Kendall's Tau correlation set 1 and set 4: 0.4257
Kendall's Tau P-value: 0.0000

Pearson correlation set 1 and set 5: 0.0784
Pearson P-value: 0.1251
Kendall's Tau correlation: 0.0516
Kendall's Tau P-value: 0.1313
```

![Query Correlation Matrices](query_correlation_matrices.png)

## 2. pyterrier bm25 results
```
The original
$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c /home/wenxin/project/data/2025/dev3-2025/qrel-first-100.txt ../bm25-2/dev3/run.txt
recip_rank              all     0.3038
recall_1000             all     0.8100
ndcg_cut_10             all     0.3302
ndcg_cut_1000           all     0.3894

llm query elicitation (google/gemini-2.5-flash-lite)
trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c /home/wenxin/project/data/2025/dev3-g1-2025/qrel.txt 
$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c /home/wenxin/project/data/2025/dev3-g1-2025/qrel.txt dev3-g1-run.txt
recip_rank              all     0.0896
recall_1000             all     0.5500
ndcg_cut_10             all     0.0999
ndcg_cut_1000           all     0.1583

llm query elicitation (openai/gpt-4o-mini)
$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c /home/wenxin/project/data/2025/dev3-g1-2025/qrel.txt dev3-o4-run.txt
recip_rank              all     0.0913
recall_1000             all     0.6100
ndcg_cut_10             all     0.0899
ndcg_cut_1000           all     0.1657
```

## 3. LLM retrieval results

Retrieval model config:
- gemini-2.5-flash
- context length: 5000
- temperature: 0

```
TREC original dev3 queries
$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c /home/wenxin/project/data/2025/dev3-2025/qrel-first-100.txt output/gmn-flash-0801/dev3-org.txt

recip_rank              all     0.4381
recall_1000             all     0.6400
ndcg_cut_10             all     0.4792
ndcg_cut_1000           all     0.4862
```

```
llm query elicitation (google/gemini-2.5-flash-lite)
$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c /home/wenxin/project/data/2025/dev3-2025/qrel-first-100.txt output/gmn-flash-0801/dev3-g1.txt

recip_rank              all     0.3797
recall_1000             all     0.6000
ndcg_cut_10             all     0.4211
ndcg_cut_1000           all     0.4311
```

```
llm query elicitation (openai/gpt-4o-mini)
$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c /home/wenxin/project/data/2025/dev3-2025/qrel-first-100.txt output/gmn-flash-0801/dev3-o4.txt
recip_rank              all     0.3860
recall_1000             all     0.6100
ndcg_cut_10             all     0.4283
ndcg_cut_1000           all     0.4382
```