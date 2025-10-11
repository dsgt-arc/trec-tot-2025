#!/usr/bin/env python3
"""
Script to test Ollama server reranking functionality.
Takes a port number (default 11434) and makes a request to rerank documents.
"""

import argparse
import requests
import sys
from typing import List, Dict, Any


def create_rerank_prompt(query: str, documents: List[str]) -> str:
    """
    Create a prompt for reranking documents based on a query.
    """
    prompt = f"""You are a document reranking system. Given a query and a list of documents, 
rank the documents in order of relevance to the query. Return only the ranking as a JSON list 
where each item contains the document index (0-based) and a relevance score (0-1).

Query: {query}

Documents:
"""
    for i, doc in enumerate(documents):
        prompt += f"{i}. {doc}\n"
    
    prompt += """
Please return a JSON response in this format:
[
    {"index": 0, "score": 0.95, "document": "document text"},
    {"index": 1, "score": 0.8, "document": "document text"},
    {"index": 2, "score": 0.3, "document": "document text"}
]

Rank from most relevant to least relevant."""
    
    return prompt


def make_ollama_request(port: int, prompt: str, model: str) -> Dict[Any, Any]:
    """
    Make a request to the Ollama server.
    """
    url = f"http://localhost:{port}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama server on port {port}")
        print("Make sure Ollama is running with: ollama serve")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error: Request timed out")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test Ollama reranking functionality")
    parser.add_argument("--port", type=int, default=11434, 
                       help="Port number for Ollama server (default: 11434)")
    parser.add_argument("--model", type=str, default="gemma3:12b",
                       help="Ollama model to use (default: llama2)")
    
    args = parser.parse_args()
    
    # Test query and documents
    query = "What is the capital of France?"
    documents = [
        "London is the capital city of England and the United Kingdom. It is situated on the River Thames in southeast England.",
        "Paris is the capital and most populous city of France. It is located in the north-central part of the country.",
        "Berlin is the capital and largest city of Germany. It is located in northeastern Germany on the banks of the rivers Spree and Havel."
    ]
    
    print(f"Testing Ollama reranking on port {args.port}")
    print(f"Using model: {args.model}")
    print(f"Query: {query}")
    print("\nOriginal documents:")
    for i, doc in enumerate(documents):
        print(f"{i+1}. {doc}")
    
    # Create the reranking prompt
    prompt = create_rerank_prompt(query, documents)
    
    print("\n" + "="*50)
    print("Making request to Ollama server...")
    print("="*50)
    
    # Make request to Ollama
    response = make_ollama_request(args.port, prompt, args.model)
    
    if 'response' in response:
        response_text = response['response']
        print(f"\nRaw Ollama Response:\n{response_text}")
    else:
        print("Error: No response field in Ollama response")
        print(f"Full response: {response}")


if __name__ == "__main__":
    main()