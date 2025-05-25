# Benchmarks for TREC-ToT (2024)

The following benchmarks (& runs) are available. Results are for the dev2 set.:


| Benchmark            | Runfiles | NDCG@10 | NDCG@1000 |  MRR@1000 |R@1000  |
|----------------------|----------|----------|-----------------|-------|----|
| [BM25](BM25.md) (k1=1, b=1.0) |  [runs](runs/bm25/) | 0.0657  |0.1033| 0.0590 | 0.3600|
| [Dense Retrieval (SBERT)](DENSE.md) (DR) |  [runs](runs/DR/) | 0.1040 | 0.1665   | 0.0901  | 0.5600| 

 
## Enviorment Setup

BM25 search is implemented using Anserini, which needs Java 21 and python 3.10.
To install Java 21, you can download the java jdk from the java offical website.

```
# To check java version
java --version

# (optional) To set java version, when you have multiple java versions installed
export JAVA_HOME=$(/usr/libexec/java_home -v 21)
```

To setup python enviroment
```
## (optional) create virtual env
pyenv install 3.10.12
pyenv virtualenv 3.10.12 trec-tot-2024
pyenv activate trec-tot-2024

## install requirements
python -m pip install --upgrade pip
pip install ir_datasets sentence-transformers
pip install pytrec_eval faiss-cpu
pip install pyserini
pip install datasets # needed to run train dense retriever
pip install 'transformers[torch]'

## checking the library versions
$ pip list
...
faiss-cpu             1.11.0
pyserini              0.44.0
...
``` 

### 2024 Dataset
1. Download 2024 dataset from https://trec-tot.github.io/guidelines-2024 (See Datasets section) or https://zenodo.org/records/11185090. (Note: the corpus is around 3GB, so it may take a while to download). Noted: the 2024 dataset corpus contains 3m wikipedia documents, while the 2025 dataset uses the whole 6m wikipedia english documents as the corpus.

Command (if you prefer)
```
wget https://zenodo.org/api/records/11185090/files-archive -O tot-2024-dataset.zip && unzip tot-2024-dataset.zip
```

2. set DATA_PATH to the folder which contains the uncompressed files s.t:

```
DATA_PATH/
  | train-2024
  | | - queries.jsonl
  | |  - qrel.txt
  | dev1-2024
  | | - queries.jsonl
  | | - qrel.txt
  | dev2-2024
  | | - queries.jsonl
  | | - qrel.txt
  | corpus.jsonl
```

Quick test to see if data is setup properly:
```
python tot.py
```
The command above should print the correct number of train/dev queries and the number of documents 
in the corpus, along with example queries and documents. The output looks like

```
trec-tot:dev2-2024
n queries: 150

trec-tot:train-2024
n queries: 150

trec-tot:dev1-2024
n queries: 150

...
corpus size:  3185450
```

## Run sparse search (BM25) baseline
Follow [BM25.md](BM25.md)

## Run dense search (BM25) baseline
Follow [DENSE.md](DENSE.md)