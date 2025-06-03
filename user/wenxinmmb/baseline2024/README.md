# Benchmarks for TREC-ToT
Adapted from https://github.com/TREC-ToT/bench/tree/main/trec24

## Environment Setup

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
pyenv virtualenv 3.10.12 trec-tot-2025
pyenv activate trec-tot-2025

## install requirements
python -m pip install --upgrade pip
pip install ir_datasets sentence-transformers
pip install pytrec_eval faiss-cpu
pip install pyserini
pip install datasets # needed to run train dense retriever
pip install 'transformers[torch]'
pip install pyarrow # needed to write parquet files

## checking the library versions
$ pip list
...
faiss-cpu             1.11.0
pyserini              0.44.0
...
``` 

### 2024 Dataset (You can skip this to download 2025 dataset directly)
1. Download 2024 dataset from https://trec-tot.github.io/guidelines-2024 (See Datasets section) or https://zenodo.org/records/11185090. (Noted: the corpus is around 3GB, so it may take a while to download. The 2024 dataset corpus contains 3m wikipedia documents, while the 2025 dataset uses the whole 6m wikipedia english documents as the corpus.)

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

3. Quick test to see if data is setup properly and build the ir_datasets index for tot-2024 data:
```
python tot_24.py
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

### 2025 Dataset
1. Download 2025 dataset from https://zenodo.org/records/15356599 

2. set DATA_PATH to the folder which contains the uncompressed files s.t:

```
DATA_PATH/
  | train-2025
  | | - queries.jsonl
  | |  - qrel.txt
  | dev1-2025
  | | - queries.jsonl
  | | - qrel.txt
  | dev2-2025
  | | - queries.jsonl
  | | - qrel.txt
  | dev3-2025
  | | - queries.jsonl
  | | - qrel.txt
  | corpus.jsonl
  | offsets.jsonl (optional)
```
*Noted: You may need to rename the files and create the directory structure to be exactly the same as shown below.

3. Run the following script build the ir_datasets index for tot-2025 data:
```
python tot_25.py
```
The command above should print the correct number of train/dev queries and the number of documents in the corpus, along with example queries and documents.
It also creates a dataset under $homedir/.ir_datasets/trec-tot25 so that tot 2025 data can be used as part of the `ir_datasets` library.

The output of the script looks like

```
trec-tot:dev2-2025
n queries: 143

trec-tot:train-2025
n queries: 143

trec-tot:dev3-2025
n queries: 536

error loading trec-tot:test-2025, skipping! <--- This is expected since the test data has not been released
trec-tot:dev1-2025
n queries: 142

...
corpus size:  6407814
```

## Run sparse search (BM25) baseline
Follow [BM25.md](BM25.md)

## Run dense search (distilbert) baseline
Follow [DENSE.md](DENSE.md)