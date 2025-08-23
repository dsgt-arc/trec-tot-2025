import pandas as pd
import pyterrier as pt
import json

# This file calculate the scores for document-query pairs from various
# retrieval methods such as pyterrier bm25

if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    index_path = "/home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index"
    with open("tmp14003441.txt", "r", encoding="utf-8") as f:
        first_line = f.readline()
        text14003441 = json.loads(first_line)["text"]

    with open("tmp73056621.txt", "r", encoding="utf-8") as f:
        first_line = f.readline()
        text73056621 = json.loads(first_line)["text"]

    df = pd.DataFrame(
    [
        ["q1", "i like movie", "14003441", text14003441],
        ["q1", "i like movie", "73056621", text73056621],
    ],
    columns=["qid", "query", "docno", "text"])

    index = pt.IndexFactory.of(index_path)
    textscorer = pt.terrier.TextScorer(takes="docs", body_attr="text", wmodel="BM25",
                               background_index=index)
    rtr = textscorer.transform(df)
    print(rtr)
    # format
    #   qid     docno                                               text  rank      score         query
    # 0  q1  14003441  The bag-of-words model is a model of text repr...     0  15.329568  i like movie
    # 1  q1  73056621  Wayne Pashley is an Australian supervising sou...     1  12.463906  i like movie

