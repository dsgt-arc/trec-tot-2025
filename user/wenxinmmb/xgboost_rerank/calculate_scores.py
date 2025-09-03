import pandas as pd
import json
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
import argparse
import os

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
            yield str(obj["query_id"]), obj["query"]

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
    parser.add_argument("--retrieval-path", required=True, help="Path to the retrieval results file.")
    parser.add_argument("--queries-path", required=True, help="Path to the queries file.")
    parser.add_argument("--corpus-path", required=True, help="Path to the corpus file.")
    parser.add_argument("--offset-path", required=True, help="Path to the corpus offset mapping file.")
    parser.add_argument("--bm25-index-path", required=True, help="Path to the PyTerrier BM25 index.")
    parser.add_argument("--output-dir", default="outputs/scores", help="Directory to save the output files. Default is 'outputs/scores'.")
    parser.add_argument("--no-reorder", action="store_true", help="Do not reorder results by score; use the original order.")
    parser.add_argument("--start-query", type=int, default=0, help="Start processing from the kth query (0-based index). Default is 0.")
    parser.add_argument("--num-queries", type=int, default=None, help="Number of queries to process from the start-query. Default is all queries.")
    return parser.parse_args()

# Example usage
if __name__ == "__main__":
    args = parse_arguments()

    queries_path = args.queries_path
    retrieval_path = args.retrieval_path
    corpus_path = args.corpus_path
    offset_path = args.offset_path
    bm25_index_path = args.bm25_index_path
    output_prefix = f"{args.output_dir}/{args.retrieval_path.split('/')[-1].split('.')[0]}--md-{args.mode}"

    queryid_to_query = load_queries(queries_path)
    queryid_to_docnos = load_trec_results(retrieval_path)
    offset_mapping = load_offset_mapping(offset_path)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "pt":
        import pyterrier as pt
        if not pt.java.started():
            pt.java.init()
        index = pt.IndexFactory.of(bm25_index_path)
        textscorer = pt.terrier.TextScorer(takes="docs", body_attr="text", wmodel="BM25", background_index=index)
        tokeniser = pt.java.autoclass(
            "org.terrier.indexing.tokenisation.Tokeniser"
        ).getTokeniser()
        model = None
    else:
        textscorer = None
        tokeniser = None
        # Initialize the BGE-M3 model
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) # add devices="cuda:0" to use GPU?

    if args.mode == "bge":
        f_dense = open(f"{output_prefix}-dense.txt", "a", encoding="utf-8")
        f_sparse = open(f"{output_prefix}-sparse.txt", "a", encoding="utf-8")
    else:
        f = open(f"{output_prefix}.txt", "a", encoding="utf-8")

    start_query = args.start_query
    if args.num_queries is not None:
        end_query = start_query + args.num_queries - 1
    
    query_iter = iter_queries(queries_path)
    trec_iter = iter_trec_results(retrieval_path)
    
    try:
        queryid, query_text = next(query_iter)
        trec_qid, docnos = next(trec_iter)
    except StopIteration:
        print("No queries or retrieval results to process.")
        exit(0)
    
    idx = 0  # Position in the query sequence (0-based)
    processed_queries = 0  # Number of queries examined from start_query onwards
    
    with tqdm(desc="Processing queries") as pbar:
        while True:
            # Skip queries that don't have retrieval results
            while queryid < trec_qid:
                print(f"Warning: No retrieval results found for query {queryid}, skipping...")
                try:
                    queryid, query_text = next(query_iter)
                except StopIteration:
                    print("Finished processing all queries.")
                    break
                idx += 1
                # Increment processed_queries if we're at or past start_query
                if idx > start_query:
                    processed_queries += 1
            
            # Check if we're done
            if queryid < trec_qid:
                break
                
            # Skip queries before start_query
            if idx < start_query:
                try:
                    queryid, query_text = next(query_iter)
                    trec_qid, docnos = next(trec_iter)
                except StopIteration:
                    break
                idx += 1
                continue

            # Increment processed_queries for queries at or after start_query
            processed_queries += 1

            # Check if we've processed enough queries
            if args.num_queries is not None and processed_queries > args.num_queries:
                break

            # Ensure queryid <= trec_qid invariant holds
            assert queryid <= trec_qid, f"Query ID invariant violated: {queryid} > {trec_qid}"
            
            if queryid == trec_qid:
                assert len(docnos) > 0, f"No document IDs found for query ID: {queryid}"
                docno_to_text = get_doc_texts(docnos, offset_mapping, corpus_path)
                docs = [
                    {"docno": docno, "text": docno_to_text.get(docno)}
                    for docno in docnos if docno_to_text.get(docno)
                ]
                if not docs:
                    # Move to next query and trec result
                    try:
                        queryid, query_text = next(query_iter)
                        trec_qid, docnos = next(trec_iter)
                    except StopIteration:
                        break
                    idx += 1
                    continue
                
                query_docs_dict = {queryid: {"query": query_text, "docs": docs}}
                scores = score_query_docs(query_docs_dict, textscorer, tokeniser, args.mode, model)

                if args.mode == "bge":
                    if not args.no_reorder:
                        docs_sorted = sorted(scores[queryid], key=lambda x: x["dense_score"], reverse=True)
                    else:
                        docs_sorted = scores[queryid]
                    for rank, doc in enumerate(docs_sorted):
                        f_dense.write(f"{queryid} Q0 {doc['docno']} {rank+1} {doc['dense_score']} {args.mode}-dense\n")

                    if not args.no_reorder:
                        docs_sorted = sorted(scores[queryid], key=lambda x: x["sparse_score"], reverse=True)
                    else:
                        docs_sorted = scores[queryid]
                    for rank, doc in enumerate(docs_sorted):
                        f_sparse.write(f"{queryid} Q0 {doc['docno']} {rank+1} {doc['sparse_score']} {args.mode}-sparse\n")

                else:
                    if not args.no_reorder:
                        docs_sorted = sorted(scores[queryid], key=lambda x: x["score"], reverse=True)
                    else:
                        docs_sorted = scores[queryid]
                    for rank, doc in enumerate(docs_sorted):
                        score = doc.get("score", 0)
                        f.write(f"{queryid} Q0 {doc['docno']} {rank+1} {score} {args.mode}\n")
                
                pbar.update(1)
            
            # Move to next query and trec result
            try:
                queryid, query_text = next(query_iter)
                trec_qid, docnos = next(trec_iter)
            except StopIteration:
                break
            idx += 1

    # Close files
    if args.mode == "bge":
        f_dense.close()
        f_sparse.close()
    else:
        f.close()

