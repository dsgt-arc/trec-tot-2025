import json
import os
from typing import List, Dict, Any
import sys

# Add the parent directory to the path to import rerank_openrouter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rerank_openrouter import SafeOpenAI

def load_queries(queries_file: str) -> Dict[str, str]:
    """Load queries from JSONL file."""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_data = json.loads(line.strip())
            queries[query_data['id']] = query_data['query']
    return queries

def load_corpus_offset(offset_file: str) -> Dict[str, int]:
    """Load corpus offset mapping."""
    offset_map = {}
    with open(offset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            doc_data = json.loads(line.strip())
            offset_map[doc_data['id']] = line_num
    return offset_map

def get_document_content(doc_id: str, corpus_file: str, offset_map: Dict[str, int]) -> Dict[str, str]:
    """Get document content from corpus file using offset."""
    if doc_id not in offset_map:
        return {"title": "", "text": ""}
    
    line_num = offset_map[doc_id]
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == line_num:
                doc_data = json.loads(line.strip())
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
        
        query_text = queries[query_id]
        candidates = []
        
        for doc_id in doc_ids:
            doc_content = get_document_content(doc_id, corpus_file, offset_map)
            candidate_text = f"{doc_content['title']} {doc_content['text']}"
            candidates.append({
                "id": doc_id,
                "text": candidate_text.strip()
            })
        
        if candidates:
            rerank_requests.append({
                "query_id": query_id,
                "query": query_text,
                "candidates": candidates
            })
    
    return rerank_requests

def batch_rerank_with_openrouter(rerank_requests: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Perform batch reranking using OpenRouter Gemini Flash 2.5."""
    
    # Initialize SafeOpenAI with OpenRouter Gemini Flash 2.5
    reranker = SafeOpenAI(
        model="google/gemini-2.0-flash-exp:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_tokens=2000
    )
    
    results = {}
    
    for request in rerank_requests:
        query_id = request["query_id"]
        query = request["query"]
        candidates = request["candidates"]
        
        print(f"Reranking query {query_id} with {len(candidates)} candidates...")
        
        # Construct rerank prompt
        prompt = f"""Given the query and a list of candidate documents, rerank the documents based on their relevance to the query. Return the documents in order from most relevant to least relevant.

Query: {query}

Candidates:
"""
        
        for i, candidate in enumerate(candidates):
            prompt += f"{i+1}. ID: {candidate['id']}\nText: {candidate['text']}\n\n"
        
        prompt += """Please rerank these documents based on their relevance to the query. Return the result as a JSON list with the document IDs in order from most relevant to least relevant, like this:
["doc_id_1", "doc_id_2", "doc_id_3", ...]"""
        
        try:
            # Get reranking from the model
            response = reranker.generate(prompt)
            
            # Parse the response to extract document IDs
            ranked_ids = json.loads(response.strip())
            
            # Create ranked results with scores
            ranked_results = []
            for rank, doc_id in enumerate(ranked_ids):
                score = len(ranked_ids) - rank  # Higher rank = higher score
                ranked_results.append({
                    "doc_id": doc_id,
                    "rank": rank + 1,
                    "score": score
                })
            
            results[query_id] = ranked_results
            
        except Exception as e:
            print(f"Error reranking query {query_id}: {e}")
            # Fallback: keep original order
            ranked_results = []
            for rank, candidate in enumerate(candidates):
                ranked_results.append({
                    "doc_id": candidate["id"],
                    "rank": rank + 1,
                    "score": len(candidates) - rank
                })
            results[query_id] = ranked_results
    
    return results

def main():
    # File paths
    run_file = "shared_retrieval_results/gemini-2.5-flash/dev1.run"
    queries_file = "2025/dev1-2025/queries.jsonl"
    corpus_file = "corpus.jsonl"
    offset_file = "corpus-offset"
    
    # Construct rerank requests
    print("Loading data and constructing rerank requests...")
    rerank_requests = construct_rerank_requests(run_file, queries_file, corpus_file, offset_file)
    print(f"Constructed {len(rerank_requests)} rerank requests")
    
    # Perform batch reranking
    print("Starting batch reranking...")
    results = batch_rerank_with_openrouter(rerank_requests)
    
    # Save results
    output_file = "reranked_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Reranking completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()