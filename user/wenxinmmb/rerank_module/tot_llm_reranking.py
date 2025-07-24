"""
TREC ToT LLM Reranking Script

This script performs neural reranking of TREC retrieval results using Large Language Models (LLMs).
It takes first-stage retrieval results (TREC run file) and reranks the documents using an LLM
to improve retrieval quality.

Key Features:
1. **Resumable Processing**: The script processes queries one at a time and maintains a checkpoint
   file to track progress. If interrupted, it can resume from where it left off without 
   reprocessing already completed queries.

2. **Dual API Support**: 
   - Local Ollama: Default mode for running local LLM models (e.g., Gemma, Llama)
   - OpenRouter: Cloud-based API access to various LLM providers when using --use-openrouter flag

3. **Incremental Output**: Results are written immediately after each query is processed,
   ensuring no data loss during long-running experiments.

4. **Configurable History**: Optional inference invocation history can be saved for debugging
   and analysis purposes using --save-invocations-history flag.

Input Requirements:
- TREC run file: First-stage retrieval results for the queries in the queries file
- Queries file: JSONL format with query_id and query text
- Corpus file: JSONL format with document content
- Offset mapping: JSON file for efficient document lookup

Queries file and corpus file can be downloaded from
https://trec-tot.github.io/guidelines

Offset mapping file `corpus-offset-mapping.json` can be downloaded from the shared Google Drive folder:
https://drive.google.com/drive/u/0/folders/1IGHLJHGxbZ

Output Files:
- rerank-results.jsonl: Reranked results in JSONL format
- rerank-results.txt: Reranked results in TREC eval format
- checkpoint.json: Progress tracking for resumable execution
- inference_invocations_history.json: Optional API call history

Usage Examples:

1. Basic usage with local Ollama:
   python tot_llm_reranking.py \
     --input-trec-run "shared_retrieval_results/gemini-2.5-flash/dev3.run" \
     --queries-file "2025/dev3-2025/queries.jsonl" \
     --corpus-file "2025/corpus.jsonl" \
     --offset-file "corpus-offset-mapping.json" \
     --output-dir "outputs/gemma3-12b-rerank"

2. Using OpenRouter with a different model:
   export OPENROUTER_API_KEY="your-api-key-here"
   python tot_llm_reranking.py \
     --input-trec-run "shared_retrieval_results/bge-passage-dense/dev3.run" \
     --queries-file "2025/dev3-2025/queries.jsonl" \
     --corpus-file "2025/corpus.jsonl" \
     --offset-file "corpus-offset-mapping.json" \
     --output-dir "outputs/gemma-3-27b-rerank" \
     --use-openrouter \
     --model "google/gemma-3-27b-it" \
     --save-invocations-history

3. Custom local model with debugging:
   python tot_llm_reranking.py \
     --input-trec-run "shared_retrieval_results/pyterrier-bm25/dev3.run" \
     --queries-file "2025/dev3-2025/queries.jsonl" \
     --corpus-file "2025/corpus.jsonl" \
     --offset-file "corpus-offset-mapping.json" \
     --output-dir "outputs/llama3-8b-rerank" \
     --model "llama3:8b" \
     --api-url "http://localhost:8080/v1" \
     --save-invocations-history
"""

import json
import os
import argparse
import datetime
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

def batch_rerank_with_openrouter(rerank_requests: List[Dict[str, Any]], 
                                  model: str, 
                                  api_base: str,
                                  api_keys: List[str],
                                  output_dir: str,
                                  save_invocations_history: bool,
                                  prompt_template_path: str = "rank_llm/src/rank_llm/rerank/prompt_templates/rank_lrl_template.yaml") -> None:
    """Perform batch reranking using openai API compatible models, processing one at a time"""

    ranker = SafeOpenaiBackend(
        model=model,
        context_size=8192,
        keys=api_keys, # API keys for the service
        api_base=api_base,
        prompt_template_path=prompt_template_path
    )
    
    # Load checkpoint if exists
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    processed_queries = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            processed_queries = set(checkpoint_data.get('processed_queries', []))
            print(f"Resuming from checkpoint. Already processed {len(processed_queries)} queries.")
    
    # Process each request individually
    for i, request in enumerate(rerank_requests):
        query_id = request.query.qid

        # Skip if already processed
        if query_id in processed_queries:
            print(f"Skipping query {query_id} (already processed)")
            continue

        print(f"Processing query {i+1}/{len(rerank_requests)}: {query_id}")

        try:
            # Process single request
            results = ranker.rerank_batch([request],
                                        populate_invocations_history=save_invocations_history,
                                        logging=False,
                                        rank_end=1000)
            
            # Write results immediately
            writer = DataWriter(results, append=True)
            writer.write_in_jsonl_format(f"{output_dir}/rerank-results.jsonl")
            writer.write_in_trec_eval_format(f"{output_dir}/rerank-results.txt")
            
            # Only write invocations history if requested
            if save_invocations_history:
                writer.write_inference_invocations_history(f"{output_dir}/inference_invocations_history.json")
            
            # Update checkpoint
            processed_queries.add(query_id)
            checkpoint_data = {
                'processed_queries': list(processed_queries),
                'last_processed_query': query_id,
                'total_processed': len(processed_queries),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
            print(f"Successfully processed query {query_id}")
            
        except Exception as e:
            print(f"Error processing query {query_id}: {str(e)}")
            # Continue with next query instead of failing completely
            continue

def main():
    parser = argparse.ArgumentParser(description='TREC ToT LLM Reranking Script')
    parser.add_argument('--input-trec-run', type=str, required=True,
                        help='Path to the TREC run file which contains the first stage retrieval results, i.e "shared_retrieval_results/gemini-2.5-flash/dev3.run"')
    parser.add_argument('--queries-file', type=str, required=True,
                        help='Path to the queries JSONL file, i.e "2025/dev3-2025/queries.jsonl"')
    parser.add_argument('--corpus-file', type=str, required=True,
                        help='Path to the corpus JSONL file, i.e "2025/dev3-2025/corpus.jsonl"')
    parser.add_argument('--offset-file', type=str, required=True,
                        help='Path to the corpus offset mapping JSON file. Download the `corpus-offset-mapping.json` file from the shared google drive data folder https://drive.google.com/drive/u/0/folders/1IGHLJHGxbZ-P4xnpTc2OpDnoWiDClYxp.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--model', type=str, default='gemma3:12b',
                        help='Model name for reranking (default: gemma3:12b)')
    parser.add_argument('--api-url', type=str, default='http://localhost:11434/v1',
                        help='API base URL (default: http://localhost:11434/v1)')
    parser.add_argument('--use-openrouter', action='store_true', default=False,
                        help='Use OpenRouter API instead of local Ollama (default: False)')
    parser.add_argument('--save-invocations-history', action='store_true', default=False,
                        help='Save inference invocations history (default: False)')
    parser.add_argument('--prompt-template-path', type=str, default="rank_llm/src/rank_llm/rerank/prompt_templates/rank_lrl_template.yaml",
                        help='Path to the prompt template YAML file (default: rank_llm/src/rank_llm/rerank/prompt_templates/rank_lrl_template.yaml)')
    
    args = parser.parse_args()

    # Configure API settings based on OpenRouter flag
    if args.use_openrouter:
        api_base = "https://openrouter.ai/api/v1"
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required when using --use-openrouter")
        api_keys = [api_key]
        print("Using OpenRouter API")
    else:
        api_base = args.api_url
        api_keys = ['ollama'] # the key is not validated for local model
        print(f"Using local API at {api_base}")

    # File paths from command line arguments
    run_file = args.input_trec_run
    queries_file = args.queries_file
    corpus_file = args.corpus_file
    offset_file = args.offset_file
    output_dir = args.output_dir

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct rerank requests
    print("Loading data and constructing rerank requests")
    rerank_requests = construct_rerank_requests(run_file, queries_file, corpus_file, offset_file)
    print(f"Constructed {len(rerank_requests)} rerank requests")

    # Perform batch reranking
    print("Starting batch reranking")
    batch_rerank_with_openrouter(rerank_requests, 
                                 args.model, 
                                 api_base,
                                 api_keys,
                                 output_dir,
                                 args.save_invocations_history,
                                 prompt_template_path=args.prompt_template_path)
    
    print("Reranking completed successfully!")
    print(f"Results saved in {output_dir}/rerank-results.jsonl and {output_dir}/rerank-results.txt")
    if args.save_invocations_history:
        print(f"Invocations history saved in {output_dir}/inference_invocations_history.json")
    print(f"Checkpoint saved in {output_dir}/checkpoint.json")

if __name__ == "__main__":
    main()