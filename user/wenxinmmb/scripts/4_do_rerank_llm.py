import json
import os
from typing import List, Dict, Any

from rank_llm.rerank.listwise.rank_openai import SafeOpenaiBackend
from rank_llm.data import Request, Query, Candidate, DataWriter

def load_queries(queries_file: str) -> Dict[str, str]:
    """Load queries from JSONL file."""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_data = json.loads(line.strip())
            queries[query_data['query_id']] = query_data['query']
    return queries

def load_corpus_offset(offset_file: str) -> Dict[str, Dict[str, int]]:
    """Load corpus offset mapping."""
    with open(offset_file, 'r', encoding='utf-8') as f:
        offset_map = json.load(f)
    return offset_map

def get_document_content(doc_id: str, corpus_file: str,
                         offset_map: Dict[str, Dict[str, int]]) -> Dict[str, str]:
    """Get document content from corpus file using byte offsets."""
    if doc_id not in offset_map:
        return {"title": "", "text": ""}
    
    offset_start = offset_map[doc_id]["offset_start"]
    offset_end = offset_map[doc_id]["offset_end"]
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        f.seek(offset_start)
        content = f.read(offset_end - offset_start)
        
        # Parse the JSON content
        doc_data = json.loads(content.strip())
        # Truncate text to first 1500 characters
        text = doc_data.get('text', '')[:1500]
        return {
            "title": doc_data.get('title', ''),
            "text": text
        }
    return {"title": "", "text": ""}

def parse_run_file(run_file: str) -> Dict[str, List[str]]:
    """Parse TREC run file to extract query-document pairs."""
    query_docs = {}
    with open(run_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = parts[0]
                doc_id = parts[2]
                
                if query_id not in query_docs:
                    query_docs[query_id] = []
                query_docs[query_id].append(doc_id)
    return query_docs

def construct_rerank_requests(
    run_file: str,
    queries_file: str,
    corpus_file: str,
    offset_file: str,
    range_start: int,
    range_end: int
) -> List[Dict[str, Any]]:
    """Construct rerank requests from run file and related data."""
    
    # Load all required data
    queries = load_queries(queries_file)
    offset_map = load_corpus_offset(offset_file)
    query_docs = parse_run_file(run_file)

    rerank_requests = []

    partial_query_docs = {k: query_docs[k] for k in list(query_docs.keys())[range_start:range_end]}

    for query_id, doc_ids in partial_query_docs.items():
        if query_id not in queries:
            print(f"Warning: Query {query_id} not found in queries file")
            continue

        req = Request(query=Query(text=queries[query_id], qid=query_id),
                candidates=[])

        for doc_id in doc_ids:
            doc_content = get_document_content(doc_id, corpus_file, offset_map)
            req.candidates.append(
                Candidate(docid=doc_id, doc={"title": doc_content['title'], "text": doc_content['text']}, score=0.0))
        if len(req.candidates) == 0:
            print(f"Warning: No candidates found for query {query_id}")
        rerank_requests.append(req)

    return rerank_requests

def batch_rerank_with_openrouter(rerank_requests: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Perform batch reranking using OpenRouter models"""

    # Initialize SafeOpenAI
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    print("Using OpenRouter API key:", openrouter_api_key)

    ranker = SafeOpenaiBackend(
        model="google/gemma-3-27b-it",
        # model="google/gemma-3-12b-it",
        context_size=8192,
        keys=[openrouter_api_key],
        api_base="https://openrouter.ai/api/v1",
        prompt_template_path="/home/wenxin/project/rank_llm/src/rank_llm/rerank/prompt_templates/rank_lrl_template.yaml" # TODO: do not hardcode this
    )
    
    results = ranker.rerank_batch(rerank_requests,
                                  populate_invocations_history=False,
                                  logging=False,
                                  rank_end=1000)
    return results

def main():
    # File paths
    split = "dev3"
    retrieval_model = 'gemini-2.5-flash'
    output_dir = "outputs/gemini-gemma-27B"
    data_path = "/home/wenxin/project/data/2025"
    range_start = 5
    range_end = 100
    range = f'{range_start}-{range_end}'

    run_file = f"/home/wenxin/project/shared_retrieval_results/{retrieval_model}/{split}.run"
    queries_file = f"{data_path}/{split}-2025/queries.jsonl"
    corpus_file = f"{data_path}/corpus.jsonl"
    offset_file = f"{data_path}/corpus-offset-mapping.json"

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct rerank requests
    print("Loading data and constructing rerank requests")
    rerank_requests = construct_rerank_requests(run_file, queries_file, corpus_file, offset_file, range_start, range_end)
    rerank_requests = rerank_requests
    print(f"Constructed {len(rerank_requests)} rerank requests")

    # Perform batch reranking
    print("Starting batch reranking")
    rerank_results = batch_rerank_with_openrouter(rerank_requests)
    
    # Save results
    print("Starting save output")
    writer = DataWriter(rerank_results)

    writer.write_in_jsonl_format(f"{output_dir}/rerank-results-{range}.jsonl")
    writer.write_in_trec_eval_format(f"{output_dir}/rerank-results-{range}.txt")
    writer.write_inference_invocations_history(
        f"{output_dir}/inference_invocations_history-{range}.json"
    )

if __name__ == "__main__":
    main()