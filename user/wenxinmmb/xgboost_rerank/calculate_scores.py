import pandas as pd
import pyterrier as pt
import json
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
import argparse

# Install
# pip install -U FlagEmbedding

def score_query_docs(query_docs_dict, textscorer, tokeniser, mode, model):
    """
    Given a dict {queryid: {"query": ..., "docs": [ {"docno": ..., "text": ...}, ... ]}}, return
    {queryid: [ {"docno": ..., "score": ..., "text": ...}, ... ]}
    Only scores the first 10 queries.

    Args:
        query_docs_dict (dict): Query-docs dictionary.
        textscorer (pyterrier.TextScorer): PyTerrier text scorer (used if mode="pt").
        tokeniser: PyTerrier tokeniser (used if mode="pt").
        mode (str): Scoring mode, either "pt" (PyTerrier) or "bge" (BGE-M3).

    Returns:
        dict: Scored results.
    """
    results = {}
    if mode == "bge":
        assert model is not None, "Model must be provided for BGE-M3 mode"

    for queryid, entry in query_docs_dict.items():
        query_text = entry["query"]
        docs = entry["docs"]

        if mode == "bge":
            scored_docs = []
            for doc in docs:
                doc_text = doc["text"]
                if not doc_text:
                    print(f'Warning: Document text is empty docid={doc["docno"]}')
                    continue

                # Split the document into paragraphs
                # paragraphs = doc_text.split('\n\n')[:3]  # Limit to 3 paragraphs to save time
                # paragraphs = doc_text.split('\n\n')

                # Prepare sentence pairs for batch processing
                # sentence_pairs = [[query_text, paragraph] for paragraph in paragraphs]

                # Version1

                # # Compute scores for all sentence pairs in a batch
                # scores = model.compute_score(
                #     sentence_pairs,
                #     max_passage_length=8192
                # )

                # # Extract sparse and dense scores
                # sparse_scores = scores['sparse']
                # dense_scores = scores['dense']

                # # Get the maximum scores
                # max_sparse_score = max(sparse_scores)
                # max_dense_score = max(dense_scores)

                # Version2
                # for sentence_pair in sentence_pairs[:3]:
                #     scores = model.compute_score(
                #         [sentence_pair],
                #         max_passage_length=8192
                #     )
                #     sparse_score = scores['sparse'][0]
                #     dense_score = scores['dense'][0]

                #     if 'max_sparse_score' not in locals():
                #         max_sparse_score = sparse_score
                #         max_dense_score = dense_score
                #     else:
                #         max_sparse_score = max(max_sparse_score, sparse_score)
                #         max_dense_score = max(max_dense_score, dense_score)

                # Version3
                scores = model.compute_score(
                    [[query_text, doc_text]],
                    max_passage_length=8192
                )
                max_sparse_score = scores['sparse'][0]
                max_dense_score = scores['dense'][0]

                scored_docs.append({
                    "docno": doc["docno"],
                    "sparse_score": max_sparse_score,
                    "dense_score": max_dense_score,
                    "text": doc_text
                })

            results[queryid] = scored_docs
        elif mode == "pt":
            tok_query_text = " ".join(tokeniser.getTokens(query_text))
            df = pd.DataFrame([
                {"qid": queryid, "query": tok_query_text, "docno": doc["docno"], "text": doc["text"]}
                for doc in docs
            ])
            rtr = textscorer.transform(df)
            results[queryid] = [
                {"docno": row["docno"], "score": row["score"], "text": row["text"]}
                for _, row in rtr.iterrows()
            ]

    return results

def load_queries(queries_path):
    queryid_to_query = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queryid_to_query[obj["query_id"]] = obj["query"]
    return queryid_to_query

def load_trec_results(trec_path):
    queryid_to_docnos = {}
    with open(trec_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid, _, docno = parts[:3]
            queryid_to_docnos.setdefault(qid, []).append(docno)
    return queryid_to_docnos

def load_offset_mapping(offset_path):
    with open(offset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_doc_texts(docnos, offset_mapping, corpus_path):
    docno_to_text = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for docno in docnos:
            offsets = offset_mapping.get(docno)
            if offsets is None:
                continue
            f.seek(offsets["offset_start"])
            line = f.readline()
            obj = json.loads(line)
            docno_to_text[docno] = obj["text"]
    return docno_to_text

def iter_queries(queries_path):
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield obj["query_id"], obj["query"]

def iter_trec_results(trec_path):
    with open(trec_path, "r", encoding="utf-8") as f:
        current_qid = None
        docnos = []
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid, _, docno = parts[:3]
            if current_qid is None:
                current_qid = qid
            if qid != current_qid:
                yield current_qid, docnos
                current_qid = qid
                docnos = []
            docnos.append(docno)
        if current_qid is not None:
            yield current_qid, docnos

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate BM25 or BGE-M3 scores for queries and documents.")
    parser.add_argument("--mode", required=True, choices=["pt", "bge"], help="Scoring mode: 'pt' for PyTerrier, 'bge' for BGE-M3.")
    return parser.parse_args()

# Example usage
if __name__ == "__main__":
    args = parse_arguments()

    DATA_PATH = "/home/wenxin/project/data"
    queries_path = f"{DATA_PATH}/2025/dev3-2025/queries-first-100.jsonl"
    # trec_path = f"{DATA_PATH}/results/dev3-100/bge.txt"
    trec_path = f"{DATA_PATH}/results/dev3-100/llm.txt"
    corpus_path = f"{DATA_PATH}/2025/corpus.jsonl"
    offset_path = f"{DATA_PATH}/2025/corpus-offset-mapping.json"
    index_path = "/home/wenxin/project/pyterrrier-index/trec-tot-2025-pyterrier-index"
    output_prefix = f"outputs/run4-llm-dev3-{args.mode}"

    queryid_to_query = load_queries(queries_path)
    queryid_to_docnos = load_trec_results(trec_path)
    offset_mapping = load_offset_mapping(offset_path)

    if args.mode == "pt":
        if not pt.java.started():
            pt.java.init()
        index = pt.IndexFactory.of(index_path)
        textscorer = pt.terrier.TextScorer(takes="docs", body_attr="text", wmodel="BM25", background_index=index)
        tokeniser = pt.java.autoclass(
            "org.terrier.indexing.tokenisation.Tokeniser"
        ).getTokeniser()
        model = None
    else:
        textscorer = None
        tokeniser = None
        # Initialize the BGE-M3 model
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # with open(output_path, "w", encoding="utf-8") as f:
    if args.mode == "bge":
        f_dense = open(f"{output_prefix}-dense.txt", "w", encoding="utf-8")
        f_sparse = open(f"{output_prefix}-sparse.txt", "w", encoding="utf-8")
    else:
        f = open(f"{output_prefix}.txt", "w", encoding="utf-8")

    for (queryid, query_text), (trec_qid, docnos) in tqdm(
        zip(iter_queries(queries_path), iter_trec_results(trec_path)),
        desc="Processing queries"
    ):
        assert queryid == trec_qid, f"Query ID mismatch: {queryid} != {trec_qid}"
        assert len(docnos) > 0, f"No document IDs found for query ID: {queryid}"
        # assert len(docnos) >= 999, f"Not enough document IDs found for query ID: {queryid}, len={len(docnos)}"
        docno_to_text = get_doc_texts(docnos, offset_mapping, corpus_path)
        docs = [
            {"docno": docno, "text": docno_to_text.get(docno)}
            for docno in docnos if docno_to_text.get(docno)
        ]
        if not docs:
            continue
        query_docs_dict = {queryid: {"query": query_text, "docs": docs}}
        scores = score_query_docs(query_docs_dict, textscorer, tokeniser, args.mode, model)
        if args.mode == "bge":
            docs_sorted = sorted(scores[queryid], key=lambda x: x["dense_score"], reverse=True)
            for rank, doc in enumerate(docs_sorted):
                f_dense.write(f"{queryid} Q0 {doc['docno']} {rank+1} {doc['dense_score']} {args.mode}-dense\n")

            docs_sorted = sorted(scores[queryid], key=lambda x: x["sparse_score"], reverse=True)
            for rank, doc in enumerate(docs_sorted):
                f_sparse.write(f"{queryid} Q0 {doc['docno']} {rank+1} {doc['sparse_score']} {args.mode}-sparse\n")

        else:
            docs_sorted = sorted(scores[queryid], key=lambda x: x["score"], reverse=True)
            for rank, doc in enumerate(docs_sorted):
                score = doc.get("score", 0)
                f.write(f"{queryid} Q0 {doc['docno']} {rank+1} {score} {args.mode}\n")

