#!/usr/bin/env python3
from pathlib import Path

import click
import pandas as pd
import pyterrier as pt
import ir_datasets

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tot_25 as tot

# Command:
# python baseline.py --output runs/bm25-2/dev1 --split dev1-2025 --data_path $DATA_PATH --index /home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index

def get_index(ir_dataset, index_directory):
    # PyTerrier needs an absolute path
    index_directory = index_directory.resolve().absolute()

    if (
        not index_directory.exists()
        or not (index_directory / "index-ir-metadata.yml").exists()
    ):
        # build the index
        indexer = pt.IterDictIndexer(
            str(index_directory), overwrite=True, meta={"docno": 100, "text": 20480}
        )

        # you can do some custom document processing here
        docs = (
            {"docno": i.id, "text": i.text}
            for i in ir_dataset.docs_iter()
        )
        indexer.index(docs)

    return pt.IndexFactory.of(str(index_directory))


def process_dataset(ir_dataset, index_directory, output_directory):
    if (output_directory / "run.txt.gz").exists():
        return

    index = get_index(ir_dataset, index_directory)
    bm25 = pt.terrier.Retriever(index, wmodel="BM25")

    # potentially do some query processing
    topics = pd.DataFrame(
        [
            {"qid": i.query_id, "query": i.query}
            for i in ir_dataset.queries_iter()
        ]
    )

    # PyTerrier needs to use pre-tokenized queries
    tokeniser = pt.java.autoclass(
        "org.terrier.indexing.tokenisation.Tokeniser"
    ).getTokeniser()

    topics["query"] = topics["query"].apply(
        lambda i: " ".join(tokeniser.getTokens(i))
    )

    run = bm25(topics)
    pt.io.write_results(run, output_directory / "run.txt.gz")


@click.command()
@click.option("--data_path", type=str, help="Path that stores the dataset")
@click.option("--split", type=str, default="train-2025", help="The split to process, e.g., train-2025, dev1-2025, dev2-2025, dev3-2025, test-2025. Multiple splits can be specified as a comma-separated list.")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--index", type=Path, required=True, help="The index directory.")
def main(data_path, split, output, index):
    tot.register(data_path)
    irds_name = "trec-tot:" + split
    dataset = ir_datasets.load(irds_name)
    print(f"loading split: {split}[irds_name={irds_name}]:\t{dataset}")
    process_dataset(dataset, index, Path(output))

if __name__ == "__main__":
    main()
