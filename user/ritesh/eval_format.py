import json

def jsonl_to_trec_runfile(jsonl_path, trec_output_path, run_id="runid1"):
    """
    Convert an aggregated JSONL file of top-k results to TREC run file format.
    Each line in the JSONL should have at least: query_id, metadata['id'], score.
    """
    # Collect results per query_id
    from collections import defaultdict
    results_by_query = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            query_id = obj["query_id"]
            doc_id = obj["metadata"]["id"]
            score = obj["score"]
            results_by_query[query_id].append((doc_id, score, obj))
    
    # Sort and write in TREC format
    with open(trec_output_path, "w", encoding="utf-8") as out:
        for query_id, docs in results_by_query.items():
            # Sort by score descending
            docs_sorted = sorted(docs, key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score, obj) in enumerate(docs_sorted, start=1):
                out.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_id}\n")
    print(f"TREC run file written to {trec_output_path}")


if __name__ == "__main__":
    jsonl_path = "/workspace/miniLM_dev3_top_k_results.jsonl"
    trec_output_path = "/workspace/miniLM_dev3_top_k_results.trec"
    jsonl_to_trec_runfile(jsonl_path, trec_output_path)