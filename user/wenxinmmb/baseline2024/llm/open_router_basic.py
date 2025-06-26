import os
import json
import requests
from tqdm import tqdm
import argparse

# Set the environment variable using command `export OPENROUTER_API_KEY=your_api_key`
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash"

PROMPT_TEMPLATE = (
    "Think about 5 possible entities that match the description below. "
    "Return a json object that contains the entity names at the end.\n\n"
    "Description: {query}"
)

def ask_llm(query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(query=query)}
        ],
        "max_tokens": 300,
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
    args = parser.parse_args()
    with open(args.input_file, "r", encoding="utf-8") as infile, \
        open(args.output_file, "a", encoding="utf-8") as outfile:
        for line in tqdm(infile.readlines(), desc="Processing queries"):
            item = json.loads(line)
            query_id = item["query_id"]
            query = item["query"]
            try:
                result = ask_llm(query)
            except Exception as e:
                result = {"error": str(e)}
            output = {
                "query_id": query_id,
                "result": result
            }
            outfile.write(json.dumps(output, ensure_ascii=False) + "\n")