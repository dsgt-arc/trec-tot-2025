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
    "openai/o4-mini",
    "openai/gpt-oss-120b",
    "google/gemini-2.5-pro"
]

# Command:
# python oprt_retrieve_score.py --input_file $DATA_PATH/dev3-2025/queries.jsonl --output_file output/dev3-gemini-2.5-flash-scored.jsonl --max_tokens 5000 --temperature 0.0

PROMPT_TEMPLATE = (
    "Identify up to 20 entities that are titles of Wikipedia pages which answer the following tip-of-the-tongue query. "
    "For each entity, provide the Wikipedia page title and a relevance score from 1-5, where:\n"
    "- 1 = Irrelevant to the query\n"
    "- 2 = Somewhat relevant\n"
    "- 3 = Moderately relevant\n"
    "- 4 = Highly relevant\n"
    "- 5 = Most relevant and directly answers the query\n\n"
    "Return a JSON object containing the entity titles and their relevance scores.\n\n"
    "TOT Query: {query}"
)

PROMPT_TEMPLATE_SINGLE = (
    "Identify the single best entity that is a title of a Wikipedia page which answers the following tip-of-the-tongue query. "
    "Provide the Wikipedia page title and a relevance score from 1-5, where:\n"
    "- 1 = Irrelevant to the query\n"
    "- 2 = Somewhat relevant\n"
    "- 3 = Moderately relevant\n"
    "- 4 = Highly relevant\n"
    "- 5 = Most relevant and directly answers the query\n\n"
    "Return a JSON object containing the best matching entity title and its relevance score.\n\n"
    "TOT Query: {query}"
)

def ask_llm(query, max_tokens, temperature, single_entity, exclude_response_format):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Choose prompt template based on single_entity flag
    prompt_template = PROMPT_TEMPLATE_SINGLE if single_entity else PROMPT_TEMPLATE
    
    # Define JSON schema based on single_entity flag
    if single_entity:
        json_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Wikipedia page title"
                },
                "relevance_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Relevance score from 1 (irrelevant) to 5 (most relevant)"
                }
            },
            "required": ["title", "relevance_score"],
            "additionalProperties": False
        }
    else:
        json_schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Wikipedia page title"
                            },
                            "score": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "Relevance score from 1 (irrelevant) to 5 (most relevant)"
                            }
                        },
                        "required": ["title", "score"],
                        "additionalProperties": False
                    },
                    "maxItems": 20
                }
            },
            "required": ["entities"],
            "additionalProperties": False
        }
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt_template.format(query=query)}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Conditionally exclude response_format
    if not exclude_response_format:
        data["response_format"] = {
            "type": "json_schema",
            "strict": True,
            "json_schema": json_schema
        }
    
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    data_out = response.json()
    content = data_out["choices"][0]["message"]["content"]
    print(f"LLM response: {content}")
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
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM response (default: 0.7)")
    parser.add_argument("--single_entity", action="store_true", help="Ask for only the single best entity match instead of up to 20 entities")
    parser.add_argument("--model", default=AVAILABLE_MODELS[0], help="Model to use for querying (default: first model in AVAILABLE_MODELS)")
    parser.add_argument("--exclude_response_format", action="store_true", help="Exclude response_format from the request (default: off)")
    args = parser.parse_args()
    
    MODEL = args.model

    # Print model parameters being used
    print(f"Using model: {MODEL}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Single entity mode: {args.single_entity}")
    print(f"Exclude response_format: {args.exclude_response_format}")
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
                result = ask_llm(query, args.max_tokens, args.temperature, args.single_entity, args.exclude_response_format)
            except Exception as e:
                result = {"error": str(e)}
            output = {
                "query_id": query_id,
                "result": result
            }
            outfile.write(json.dumps(output, ensure_ascii=False) + "\n")