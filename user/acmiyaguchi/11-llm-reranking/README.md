# llm reranking

This directory contains supplementary code for wenxin's reranking code.
I'm setting it up so that it's relatively straightforward to run the code.

First make sure you have a copy of https://zenodo.org/records/15356599 locally.
For reranking I only need the queries and qrels, but I've included everything for completeness.

```bash
rclone sync $HOME/scratch/trec-tot-2025/results/rerank gdrive-trec-tot-2025:data/rerank
```

## v2 experiments

Starting to get methodical, but still learning how to do things

```bash
# about 12 hours
MODEL=google/gemma-3-12b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=gaunernst/gemma-3-12b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=gaunernst/gemma-3-27b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch

# let's see if we can use the quantized models

# about 3 hours 282944
MODEL=gaunernst/gemma-3-1b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
# time 2:33 283276, 0.0696
MODEL=gaunernst/gemma-3-4b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
# about 8 hours failed in 20 minutes 283277,
MODEL=gaunernst/gemma-3-12b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch

# lets see what happens on the other datasets

# 283351
MODEL=gaunernst/gemma-3-12b-it-qat-compressed-tensors RETRIEVAL_MODEL=bge-passage-dense DEVSET=dev3 sbatch rerank.sbatch

# 10 minutes, 283372, ndcg 32 -> 9, qat is pretty bad
MODEL=gaunernst/gemma-3-12b-it-qat-compressed-tensors RETRIEVAL_MODEL=gemini-2.5-flash DEVSET=dev3 sbatch rerank.sbatch

# what happens when we use the bigger model? 285322 -> 0.0939
MODEL=google/gemma-3-12b-it RETRIEVAL_MODEL=gemini-2.5-flash DEVSET=dev3 sbatch rerank.sbatch

# limit the number of ranked items? 284740, about 1 hour, 0.0929
MODEL=google/gemma-3-12b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch --batch-rank-end 100
# using an unsloth model, 285201, 1 hour, 0.0895 ndcg@10, 20 minute difference
MODEL=unsloth/gemma-3-12b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch --batch-rank-end 100
```

It looks like the issue as not using a qrel that had the first 100 documents.
Let's generate a script to rewrite those results

```
trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/official/dev3-2025-qrel-h100.txt /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/gdrive/data/shared_retrieval_results/pyterrier-bm25/dev3.run
recip_rank              all     0.3038
recall_1000             all     0.8100
ndcg_cut_10             all     0.3302
ndcg_cut_1000           all     0.3894
```

```bash
for path in /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v2*; do
    echo $path
    trec_eval \
        -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c \
        /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/official/dev3-2025-qrel-h100.txt \
        $path/rerank-results-reformatted.txt
done
```

| Model                                | recip_rank | recall_1000 | ndcg_cut_10 | ndcg_cut_1000 |
| :----------------------------------- | ---------: | ----------: | ----------: | ------------: |
| pyterrier-bm25 (Baseline)            |     0.3038 |        0.81 |      0.3302 |        0.3894 |
| gemini-2.5-flash-rerank              |     0.3682 |        0.43 |      0.3812 |        0.3828 |
| google-gemma-3-12b-it-rerank (Set A) |     0.3698 |        0.43 |      0.3851 |        0.3851 |
| gaunernst-gemma-3-1b-it-rerank       |     0.0868 |        0.81 |      0.0955 |        0.1759 |
| gaunernst-gemma-3-27b-it-rerank      |     0.1531 |        0.27 |      0.1672 |        0.1753 |
| gaunernst-gemma-3-4b-it-rerank       |     0.3459 |        0.81 |      0.3733 |        0.4152 |
| google-gemma-3-12b-it-rerank (Set B) |     0.4745 |        0.81 |      0.4978 |        0.5262 |
| unsloth-gemma-3-12b-it-rerank        |     0.4537 |        0.81 |      0.4799 |        0.5096 |

Below is the raw data

```
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/gdrive/data/shared_retrieval_results/pyterrier-bm25/dev3.run
recip_rank              all     0.3038
recall_1000             all     0.8100
ndcg_cut_10             all     0.3302
ndcg_cut_1000           all     0.3894
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v2-gemini-2.5-flash-dev3-gaunernst-gemma-3-12b-it-qat-compressed-tensors-rerank
recip_rank              all     0.3682
recall_1000             all     0.4300
ndcg_cut_10             all     0.3812
ndcg_cut_1000           all     0.3828
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v2-gemini-2.5-flash-dev3-google-gemma-3-12b-it-rerank
recip_rank              all     0.3698
recall_1000             all     0.4300
ndcg_cut_10             all     0.3851
ndcg_cut_1000           all     0.3851
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v2-pyterrier-bm25-dev3-gaunernst-gemma-3-1b-it-qat-compressed-tensors-rerank
recip_rank              all     0.0868
recall_1000             all     0.8100
ndcg_cut_10             all     0.0955
ndcg_cut_1000           all     0.1759
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v2-pyterrier-bm25-dev3-gaunernst-gemma-3-27b-it-qat-compressed-tensors-rerank
recip_rank              all     0.1531
recall_1000             all     0.2700
ndcg_cut_10             all     0.1672
ndcg_cut_1000           all     0.1753
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v2-pyterrier-bm25-dev3-gaunernst-gemma-3-4b-it-qat-compressed-tensors-rerank
recip_rank              all     0.3459
recall_1000             all     0.8100
ndcg_cut_10             all     0.3733
ndcg_cut_1000           all     0.4152
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v2-pyterrier-bm25-dev3-google-gemma-3-12b-it-rerank
recip_rank              all     0.4745
recall_1000             all     0.8100
ndcg_cut_10             all     0.4978
ndcg_cut_1000           all     0.5262
/storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v2-pyterrier-bm25-dev3-unsloth-gemma-3-12b-it-rerank
recip_rank              all     0.4537
recall_1000             all     0.8100
ndcg_cut_10             all     0.4799
ndcg_cut_1000           all     0.5096
```

## v3 experiments

Figured out the method, using 100 items, with top 100 rank. Takes about an hour to run an experiment.

```bash
MODEL=gaunernst/gemma-3-1b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=gaunernst/gemma-3-4b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=gaunernst/gemma-3-12b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=gaunernst/gemma-3-27b-it-qat-compressed-tensors RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch

MODEL=google/gemma-3-1b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=google/gemma-3-4b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=google/gemma-3-12b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=unsloth/gemma-3-1b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=unsloth/gemma-3-4b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=unsloth/gemma-3-12b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch

# and then lets go over the combinations of bge/gemini/o4
for model in google/gemma-3-12b-it gaunernst/gemma-3-12b-it-qat-compressed-tensors gaunernst/gemma-3-27b-it-qat-compressed-tensors; do
    for retrieval in bge-passage-dense gemini-2.5-flash mini-o4; do
        MODEL=$model RETRIEVAL_MODEL=$retrieval DEVSET=dev3 sbatch rerank.sbatch
    done
done

for model in unsloth/Qwen3-8B unsloth/phi-4 mistralai/Mistral-Nemo-Instruct-2407 unsloth/Meta-Llama-3.1-8B-Instruct openai/gpt-oss-20b; do
    MODEL=$model RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch -J $model rerank.sbatch
done;

# these ones are not doable
MODEL=google/gemma-3-27b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
MODEL=unsloth/gemma-3-27b-it RETRIEVAL_MODEL=pyterrier-bm25 DEVSET=dev3 sbatch rerank.sbatch
```

```bash
for path in /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v3*; do echo $path; done

# look at log lines
for path in $(ls -1 /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v3*/*reformatted.txt | sort); do
    grep -l $path logs/* 2>/dev/null | xargs ls -t | head -1 | xargs tail -n40
done > v3.txt

# compute results directly
for path in /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v3*; do
    echo $path
    trec_eval \
        -m ndcg_cut.10,100,1000 -m recall.10,100,1000 -m recip_rank -c \
        /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/data/official/dev3-2025-qrel-h100.txt \
        $path/rerank-results-reformatted.txt
done
```

## v4 experiments

how long does it take to do varying numbers of results (up to 1000).

```bash
for model in gaunernst/gemma-3-12b-it-qat-compressed-tensors gaunernst/gemma-3-27b-it-qat-compressed-tensors; do
    for retrieval in pyterrier-bm25; do
        for n in $(seq 100 100 1000); do
            echo sleep 10
            echo MODEL=$model RETRIEVAL_MODEL=$retrieval DEVSET=dev3 BATCH_RANK_END=$n NUM_QUERIES=10 sbatch -J n$n-$model rerank.sbatch
        done
    done
done
```

```bash
# oops i messed up, they didn't get differentiated
for path in /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v4*; do echo $path; done
```

```
$ duckdb -c "
> -- Load the data
> CREATE TABLE timing AS SELECT * FROM read_csv_auto('timing_data.csv');
>
> -- Show all data
> SELECT * FROM timing ORDER BY model, query_count;
>
> -- Filter out anomalous entries (< 5 minutes runtime for 800+ queries)
> CREATE TABLE clean_timing AS
> SELECT * FROM timing
> WHERE NOT (query_count >= 800 AND walltime_minutes < 5);
>
> -- Show cleaned data
> SELECT 'CLEANED DATA:' as note;
> SELECT * FROM clean_timing ORDER BY model, query_count;
>
> -- Calculate linear regression for each model
> SELECT
>     model,
>     REGR_SLOPE(walltime_minutes, query_count) as slope_min_per_query,
>     REGR_INTERCEPT(walltime_minutes, query_count) as intercept_minutes,
>     REGR_R2(walltime_minutes, query_count) as r_squared,
>     COUNT(*) as data_points
> FROM clean_timing
> GROUP BY model;
> "
┌─────────┬─────────────┬──────────────────┬───────────┐
│  model  │ query_count │ walltime_minutes │  status   │
│ varchar │    int64    │      double      │  varchar  │
├─────────┼─────────────┼──────────────────┼───────────┤
│ 12B-QAT │         100 │             10.8 │ COMPLETED │
│ 12B-QAT │         200 │            14.93 │ COMPLETED │
│ 12B-QAT │         300 │            18.92 │ COMPLETED │
│ 12B-QAT │         400 │            22.85 │ COMPLETED │
│ 12B-QAT │         500 │            26.82 │ COMPLETED │
│ 12B-QAT │         600 │             31.4 │ COMPLETED │
│ 12B-QAT │         700 │             34.9 │ COMPLETED │
│ 12B-QAT │         800 │              1.9 │ COMPLETED │
│ 12B-QAT │         900 │             1.35 │ COMPLETED │
│ 12B-QAT │        1000 │            47.48 │ COMPLETED │
│ 27B-QAT │         100 │            13.73 │ COMPLETED │
│ 27B-QAT │         200 │            18.75 │ COMPLETED │
│ 27B-QAT │         300 │            26.78 │ COMPLETED │
│ 27B-QAT │         400 │             34.7 │ COMPLETED │
│ 27B-QAT │         500 │            42.22 │ COMPLETED │
│ 27B-QAT │         600 │             6.15 │ COMPLETED │
│ 27B-QAT │         700 │            51.52 │ COMPLETED │
│ 27B-QAT │         800 │            58.52 │ COMPLETED │
│ 27B-QAT │         900 │            65.28 │ COMPLETED │
│ 27B-QAT │        1000 │            64.62 │ COMPLETED │
├─────────┴─────────────┴──────────────────┴───────────┤
│ 20 rows                                    4 columns │
└──────────────────────────────────────────────────────┘
┌───────────────┐
│     note      │
│    varchar    │
├───────────────┤
│ CLEANED DATA: │
└───────────────┘
┌─────────┬─────────────┬──────────────────┬───────────┐
│  model  │ query_count │ walltime_minutes │  status   │
│ varchar │    int64    │      double      │  varchar  │
├─────────┼─────────────┼──────────────────┼───────────┤
│ 12B-QAT │         100 │             10.8 │ COMPLETED │
│ 12B-QAT │         200 │            14.93 │ COMPLETED │
│ 12B-QAT │         300 │            18.92 │ COMPLETED │
│ 12B-QAT │         400 │            22.85 │ COMPLETED │
│ 12B-QAT │         500 │            26.82 │ COMPLETED │
│ 12B-QAT │         600 │             31.4 │ COMPLETED │
│ 12B-QAT │         700 │             34.9 │ COMPLETED │
│ 12B-QAT │        1000 │            47.48 │ COMPLETED │
│ 27B-QAT │         100 │            13.73 │ COMPLETED │
│ 27B-QAT │         200 │            18.75 │ COMPLETED │
│ 27B-QAT │         300 │            26.78 │ COMPLETED │
│ 27B-QAT │         400 │             34.7 │ COMPLETED │
│ 27B-QAT │         500 │            42.22 │ COMPLETED │
│ 27B-QAT │         600 │             6.15 │ COMPLETED │
│ 27B-QAT │         700 │            51.52 │ COMPLETED │
│ 27B-QAT │         800 │            58.52 │ COMPLETED │
│ 27B-QAT │         900 │            65.28 │ COMPLETED │
│ 27B-QAT │        1000 │            64.62 │ COMPLETED │
├─────────┴─────────────┴──────────────────┴───────────┤
│ 18 rows                                    4 columns │
└──────────────────────────────────────────────────────┘
┌─────────┬─────────────────────┬────────────────────┬────────────────────┬─────────────┐
│  model  │ slope_min_per_query │ intercept_minutes  │     r_squared      │ data_points │
│ varchar │       double        │       double       │       double       │    int64    │
├─────────┼─────────────────────┼────────────────────┼────────────────────┼─────────────┤
│ 27B-QAT │ 0.05798848484848485 │  6.333333333333336 │ 0.6629267830691583 │          10 │
│ 12B-QAT │ 0.04066302521008403 │ 6.6975630252100835 │  0.999752265889027 │           8 │
└─────────┴─────────────────────┴────────────────────┴────────────────────┴─────────────┘
```

## v5

```bash
for model in gaunernst/gemma-3-27b-it-qat-compressed-tensors; do
    for retrieval in pyterrier-bm25; do
        for n in $(seq 100 100 1000); do
            echo sleep 10
            echo MODEL=$model RETRIEVAL_MODEL=$retrieval DEVSET=dev3 BATCH_RANK_END=$n NUM_QUERIES=10 sbatch -J n$n-$model rerank.sbatch
        done
    done
done
```

```bash
for path in /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v5*; do echo $path; done


for path in $(ls -1 /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v5*/*reformatted.txt | sort); do
    grep -l $path logs/* 2>/dev/null | xargs ls -t | head -1 | xargs tail -n40
done > v5.txt
```

## v6

Okay now I want to run things with a batch_rank_end set to 100.
I'll do this on all the dev sets.

```bash
for model in gaunernst/gemma-3-27b-it-qat-compressed-tensors; do
    for retrieval in pyterrier-bm25 bge-passage-dense gemini-2.5-flash-v2; do
        for devset in dev1 dev2 dev3; do
            echo sleep 10
            echo MODEL=$model RETRIEVAL_MODEL=$retrieval DEVSET=$devset sbatch -J $retrieval-$devset rerank.sbatch
        done
    done
done

for model in gaunernst/gemma-3-27b-it-qat-compressed-tensors; do
    for retrieval in gemini-2.5-flash-v2; do
        for devset in dev1 dev2 dev3; do
            echo sleep 10
            echo MODEL=$model RETRIEVAL_MODEL=$retrieval DEVSET=$devset sbatch -J $retrieval-$devset rerank.sbatch
        done
    done
done
```

```bash
for path in /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v6*; do echo $path; done

for path in $(ls -1 /storage/home/hcoda1/8/amiyaguchi3/scratch/trec-tot-2025/results/rerank/v6*/*reformatted.txt | sort); do
    grep -l $path logs/* 2>/dev/null | xargs ls -t | head -1 | xargs tail -n40
done > v6.txt
```
