import os
import json
import requests
from tqdm import tqdm
import argparse

# Set the environment variable using command `export OPENROUTER_API_KEY=your_api_key`
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
API_URL = "https://openrouter.ai/api/v1/chat/completions"
AVAILABLE_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemma-3-27b-it",
    "openai/o4-mini"
]
MODEL = AVAILABLE_MODELS[0]  # Default model

# Command:
# python open_router_basic.py --input_file $DATA_PATH/dev3-2025/queries.jsonl --output_file output/dev3-o4-mini.jsonl

PROMPT_TEMPLATE = (
    "Think about 5 possible entities that match the description below. "
    "Return a json object that contains the entity names at the end.\n\n"
    "Description: {query}"
)

def ask_llm(query, max_tokens=5000, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(query=query)}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "strict": True,
            "json_schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["entities"]
            }
        }
    }
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    data_out = response.json()
    print(f"LLM response: {data_out}")
    content = data_out["choices"][0]["message"]["content"]
    # Try to extract the JSON object from the response
    try:
        # If the model returns text before/after the JSON, extract the JSON part
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        print(f"Extracted JSON string: {json_str}")
        return json.loads(json_str)
    except Exception:
        raise ValueError(f"Could not parse JSON from LLM response: {content}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query LLM for entity extraction.")
    parser.add_argument("--input_file", required=True, help="Path to input queries.jsonl")
    parser.add_argument("--output_file", required=True, help="Path to output answers.jsonl")
    parser.add_argument("--max_lines", type=int, default=None, help="Maximum number of lines to process (default: process all lines)")
    parser.add_argument("--start_line", type=int, default=0, help="Starting line number to process from (default: 0)")
    parser.add_argument("--max_tokens", type=int, default=5000, help="Maximum tokens for LLM response (default: 5000)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for LLM response (default: 0.7)")
    args = parser.parse_args()
    
    # Print model parameters being used
    print(f"Using model: {MODEL}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Start line: {args.start_line}")
    if args.max_lines:
        print(f"Max lines to process: {args.max_lines}")
    print("-" * 50)
    
    with open(args.input_file, "r", encoding="utf-8") as infile, \
        open(args.output_file, "a", encoding="utf-8") as outfile:
        lines = infile.readlines()
        lines = lines[args.start_line:]  # Start from the specified line
        if args.max_lines is not None:
            lines = lines[:args.max_lines]
        for line in tqdm(lines, desc="Processing queries"):
            item = json.loads(line)
            query_id = item["query_id"]
            query = item["query"]
            try:
                result = ask_llm(query, max_tokens=args.max_tokens, temperature=args.temperature)
            except Exception as e:
                result = {"error": str(e)}
            output = {
                "query_id": query_id,
                "result": result
            }
            outfile.write(json.dumps(output, ensure_ascii=False) + "\n")