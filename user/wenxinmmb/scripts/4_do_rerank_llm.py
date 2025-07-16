import json
import os
from typing import List, Dict, Any
import sys

from rank_llm.rerank.listwise.rank_openai import SafeOpenaiBackend
from rank_llm.rerank.rankllm import PromptMode
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
        content = f.read(offset_end - offset_start) # TODO: check if this is one off.
        
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
    offset_file: str
) -> List[Dict[str, Any]]:
    """Construct rerank requests from run file and related data."""
    
    # Load all required data
    queries = load_queries(queries_file)
    offset_map = load_corpus_offset(offset_file)
    query_docs = parse_run_file(run_file)
    
    rerank_requests = []
    
    for query_id, doc_ids in query_docs.items():
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
        context_size=8192,
        keys=[openrouter_api_key],
        api_base="https://openrouter.ai/api/v1",
        prompt_template_path="/home/wenxin/project/rank_llm/src/rank_llm/rerank/prompt_templates/rank_lrl_template.yaml" # TODO: do not hardcode this
    )
    
    results = ranker.rerank_batch(rerank_requests,
                                  populate_invocations_history=True,
                                  logging=True)
    print("Ranking results:")
    for result in results:
        print(f"Query: {result.query}")
        for candidate in result.candidates:
            print(f"  Candidate ID: {candidate.docid}, Score: {candidate.score}, Text: {candidate.doc}")
    return results

def main():
    # File paths
    split = "dev3"
    data_path = "/home/wenxin/project/data/2025"
    run_file = f"/home/wenxin/project/shared_retrieval_results/gemini-2.5-flash/{split}.run"
    queries_file = f"{data_path}/{split}-2025/queries.jsonl"
    corpus_file = f"{data_path}/corpus.jsonl"
    offset_file = f"{data_path}/corpus-offset-mapping.json"
    
    # Construct rerank requests
    print("Loading data and constructing rerank requests")
    rerank_requests = construct_rerank_requests(run_file, queries_file, corpus_file, offset_file)
    rerank_requests = rerank_requests[3:5]  # TODO: remove, Limit requests for testing
    print(f"Constructed {len(rerank_requests)} rerank requests")

    print("Example rerank requests:")
    for req in rerank_requests:
        print(f"Query ID: {req.query.qid}, Candidates: {[c.docid for c in req.candidates]}")
        # print the title and text of the first candidate
        if req.candidates:
            for candidate in req.candidates[:5]:
                print(f"  Candidate - ID: {candidate.docid}, Title: {candidate.doc.get('title', '')}, Text: {candidate.doc.get('text', '')[:500]}...")
        else:
            print("  No candidates available for this query.")

    # Perform batch reranking
    print("Starting batch reranking")
    rerank_results = batch_rerank_with_openrouter(rerank_requests)
    
    # Save results
    print("Starting save output")
    writer = DataWriter(rerank_results)
    # create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    writer.write_in_jsonl_format(f"outputs/rerank_results.jsonl")
    writer.write_in_trec_eval_format(f"outputs/rerank_results.txt")
    writer.write_inference_invocations_history(
        f"outputs/inference_invocations_history.json"
    )

if __name__ == "__main__":
    main()