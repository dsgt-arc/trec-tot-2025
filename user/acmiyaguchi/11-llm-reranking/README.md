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
        -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c \
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
