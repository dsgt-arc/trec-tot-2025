# PyTerrier BM25 Baseline for TREC-ToT 2025

This directory contains a BM25 baseline implemented in [PyTerrier](https://github.com/terrier-org/pyterrier) for the 2025 edition of the [TREC Tip-of-the-Tongue (ToT) Track](https://trec-tot.github.io/). This baseline tracks the experiments in the [ir_metadata format](https://www.ir-metadata.org/) (including resource consumption for GPU/CPU/RAM and used energy) with the [TIREx tracker](https://github.com/tira-io/tirex-tracker).

## Existing Runs

The runs for all splits are available:

| ir_dataset          | run                                            |
|---------------------|------------------------------------------------|
| trec-tot/2025/train | [runs/train/run.txt.gz](runs/train/run.txt.gz) |
| trec-tot/2025/dev1  | [runs/dev1/run.txt.gz](runs/dev1/run.txt.gz)   |
| trec-tot/2025/dev2  | [runs/dev2/run.txt.gz](runs/dev2/run.txt.gz)   |
| trec-tot/2025/dev3  | [runs/dev3/run.txt.gz](runs/dev3/run.txt.gz)   |

## Existing Indices

A pre-built PyTerrier index is available online so that you can make faster experimentation:

| Index | Size | md5 |
|-------|------|-----|
|[trec-tot-2025-pyterrier-index.zip](https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-pyterrier-index.zip) | 11GB | a9a22ed35abb6cea842a7c5734987c82 |

You can download and extract this index if you want to re-run or modify this approach:

```
wget https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-pyterrier-index.zip
# md5 should be a9a22ed35abb6cea842a7c5734987c82
md5sum trec-tot-2025-pyterrier-index.zip
unzip trec-tot-2025-pyterrier-index.zip
```

## Additional dependencies
```
pip install click python-terrier==0.13.0 tirex-tracker>=0.2.14
```

## Run it locally:
```
$ python baseline.py --output runs/llm-set1-train --split generated-queries/llm-set1/train --data_path /home/wenxin/project/data/2025 --index /home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index

$ gzip -d runs/bm25-2/run.txt.gz

$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c $DATA_PATH/generated-queries/llm-set1/dev/qrel.txt set1-dev.txt
```
