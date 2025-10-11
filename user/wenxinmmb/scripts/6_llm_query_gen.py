"""
Refined LLM Query Generation Module
Extracted and refined from generate_gpt_queries.py for running generate_single function

Example:
python 6_llm_query_gen.py --use-generic --start 10 --end 100 --paragraphs-file outputs/generated_query/openai_gpt-4o-mini/json_files/

"""
import argparse
import csv
import json
import logging
import os
import requests
from datetime import datetime
from openai import OpenAI
import pandas as pd

from templates import (
    get_template, 
    get_system_message, 
    get_content_type, 
    list_available_topics
)

# Configuration
MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash-lite", 
    "openai/gpt-4o-2024-08-06"
]
MODEL = MODELS[0]  # Default to first model

# Maximum input text length for LLM processing (characters)
MAX_INPUT_LENGTH = 58000

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Global list to track warnings
warning_records = []

# Global list to track failed entries
failed_records = []

def log_failed_entry(query_id, entity_id, title, error_message):
    """
    Logs a failed entry for later output to TSV file.
    
    Args:
        query_id (str): Query ID
        entity_id (str): Entity ID
        title (str): Entity title
        error_message (str): Error message describing the failure
    """
    failed_record = {
        'query_id': query_id or '',
        'entity_id': entity_id or '',
        'title': title or '',
        'error_message': error_message
    }
    failed_records.append(failed_record)

def save_failed_entries_to_file(output_file_path):
    """
    Saves all failed entries to a TSV file.
    
    Args:
        output_file_path (str): Path to save the failed entries TSV file
    """
    if not failed_records:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.isfile(output_file_path)
    
    with open(output_file_path, "a", encoding="utf-8", newline="") as outfile:
        fieldnames = ["query_id", "entity_id", "title", "error_message"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
            
        for record in failed_records:
            writer.writerow(record)

def log_warning_to_file(warning_type, message, target_object=None, entity_id=None, query_id=None):
    """
    Logs a warning and tracks it for later output to TSV file.
    
    Args:
        warning_type (str): Type of warning
        message (str): Warning message
        target_object (str, optional): Target object name
        entity_id (str, optional): Entity ID
        query_id (str, optional): Query ID
    """
    logging.warning(message)
    
    warning_record = {
        'query_id': query_id or '',
        'entity_id': entity_id or '',
        'target_object': target_object or '',
        'warning_type': warning_type,
        'message': message
    }
    warning_records.append(warning_record)

def save_warnings_to_file(output_file_path):
    """
    Saves all collected warnings to a TSV file.
    
    Args:
        output_file_path (str): Path to save the warnings TSV file
    """
    if not warning_records:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.isfile(output_file_path)
    
    with open(output_file_path, "a", encoding="utf-8", newline="") as outfile:
        fieldnames = ["query_id", "entity_id", "target_object", "warning_type", "message"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
            
        for record in warning_records:
            writer.writerow(record)

def map_category_to_topic(category):
    """
    Maps TSV category to topic format used by templates.
    
    Args:
        category (str): Category from TSV ("person", "place", "film")
        
    Returns:
        str: Topic format ("celebrity", "landmark", "movie")
    """
    category_mapping = {
        "person": "celebrity",
        "place": "landmark", 
        "film": "movie"
    }
    
    mapped_topic = category_mapping.get(category.lower())
    if not mapped_topic:
        raise ValueError(f"Unknown category '{category}'. Valid categories: {list(category_mapping.keys())}")
    
    return mapped_topic


def read_entities_from_tsv(tsv_file_path, use_generic=False):
    """
    Reads entities from the TSV file.
    
    Args:
        tsv_file_path (str): Path to the TSV file
        use_generic (bool): If True, use 'generic' topic for all entities instead of mapping from category
        
    Returns:
        list: List of dictionaries containing entity information
    """
    try:
        df = pd.read_csv(tsv_file_path, sep='\t')
        entities = []
        
        for _, row in df.iterrows():
            if use_generic:
                topic = 'generic'
            else:
                topic = map_category_to_topic(row['category'])
                
            entity = {
                'query_id': row['query_id'],
                'entity_id': row['entity_id'],
                'title': row['title'],
                'category': row['category'],
                'url': row['url'],
                'topic': topic
            }
            entities.append(entity)
            
        return entities
        
    except Exception as e:
        raise Exception(f"Failed to read TSV file '{tsv_file_path}': {e}")


def fetch_document_from_wiki_full(target_object):
    """
    Fetches the full Wikipedia content for a given target object.
    
    Args:
        target_object (str): The name of the Wikipedia page to fetch
        
    Returns:
        str: The full text content of the Wikipedia page or error message
    """
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&titles={target_object}&explaintext=true"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        if not pages:
            return "No pages found in the API response."
        
        page = next(iter(pages.values()), None)
        if not page or "extract" not in page:
            return "Failed to retrieve content from the page."
        
        return page["extract"].strip()
        
    except requests.RequestException as e:
        return f"Failed to retrieve content from Wikipedia: {e}"


def split_document(document, max_paragraphs=None):
    """
    Splits a document into paragraphs and filters out empty ones.
    
    Args:
        document (str): The document text to split
        max_paragraphs (int, optional): Maximum number of paragraphs to return
        
    Returns:
        list: List of non-empty paragraph strings
    """
    if not document:
        return []
    
    paragraphs = document.split("\n")
    filtered_paragraphs = [para.strip() for para in paragraphs if para.strip()]
    
    if max_paragraphs:
        return filtered_paragraphs[:max_paragraphs]
    
    return filtered_paragraphs


def read_paragraphs_from_file(paragraphs_dir, target_object):
    """
    Reads paragraphs from a JSON file in the specified directory.
    
    Args:
        paragraphs_dir (str): Directory containing the JSON files
        target_object (str): The name of the target object (used to generate filename)
        
    Returns:
        list: List of paragraph strings from the JSON file
        
    Raises:
        Exception: If file cannot be read or doesn't exist
    """
    try:
        json_filename = generate_json_filename(target_object)
        file_path = os.path.join(paragraphs_dir, json_filename)
        
        if not os.path.exists(file_path):
            raise Exception(f"Paragraphs JSON file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'paragraphs' not in data:
            raise Exception(f"'paragraphs' field not found in JSON file: {file_path}")
        
        paragraphs = data['paragraphs']
        
        if not isinstance(paragraphs, list):
            raise Exception(f"'paragraphs' field must be a list in JSON file: {file_path}")
        
        if not paragraphs:
            raise Exception(f"No paragraphs found in JSON file: {file_path}")
            
        return paragraphs
        
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON format in file '{file_path}': {e}")
    except Exception as e:
        raise Exception(f"Failed to read paragraphs from '{file_path}': {e}")


def generate_json_filename(target_object):
    """
    Generates a JSON filename from the target object name.
    
    Args:
        target_object (str): The name of the target object
        
    Returns:
        str: Filename with .json extension
    """
    return f'{target_object.replace(" ", "_").replace("/", "_")}.json'


def extract_code_block_content(response, target_object=None, entity_id=None, query_id=None):
    """
    Extracts content from the first code block in the response.
    
    Args:
        response (str): The response text that may contain code blocks
        target_object (str, optional): Target object name for warning tracking
        entity_id (str, optional): Entity ID for warning tracking
        query_id (str, optional): Query ID for warning tracking
        
    Returns:
        str: Content from the first code block, or the whole response if no code block found
    """
    import re
    
    # Pattern to match code blocks (both ``` and ```)
    code_block_pattern = r'```(?:[a-zA-Z]*\n)?(.*?)```|`([^`]+)`'
    
    match = re.search(code_block_pattern, response, re.DOTALL)
    
    if match:
        # Get the content from either group (triple backticks or single backticks)
        code_content = match.group(1) if match.group(1) is not None else match.group(2)
        
        if code_content and code_content.strip():
            print("Found code block content, using it as post content")
            return code_content.strip()
        else:
            log_warning_to_file(
                "empty_code_block",
                "Code block found but is empty, using whole response as post content",
                target_object, entity_id, query_id
            )
            return response
    else:
        log_warning_to_file(
            "no_code_block",
            "No code block found in response, using whole response as post content",
            target_object, entity_id, query_id
        )
        return response


def summarize_text(text, content_type="movie"):
    """
    Summarizes the given text into two paragraphs using the LLM.
    
    Args:
        text (str): The text to summarize
        content_type (str): Type of content - "movie", "person", or "place"
        
    Returns:
        str: The summarized text
    """
    if isinstance(text, list):
        input_text = "\n\n".join(text)
    else:
        input_text = text
    
    # Capped the input length to be less than MAX_INPUT_LENGTH characters
    if len(input_text) > MAX_INPUT_LENGTH:
        input_text = input_text[:MAX_INPUT_LENGTH]
        log_warning_to_file(
            "input_length_exceeded",
            f"Input text length exceeded {MAX_INPUT_LENGTH} characters, truncating to fit",
            content_type=content_type
        )
    
    # Create appropriate prompt based on content type
    if content_type == "person":
        prompt = f"Please summarize the following description about a person into two paragraphs:\n\n{input_text}."
    elif content_type == "place":
        prompt = f"Please summarize the following description about a place into two paragraphs:\n\n{input_text}."
    elif content_type == "movie":
        prompt = f"Please summarize the following description about a movie into two paragraphs:\n\n{input_text}. Please focus on the plots, and ignore the director and actor names."
    else:
        # generic 
        prompt = f"Please summarize the following description into two paragraphs:\n\n{input_text}."
    messages = [
        {"role": "system", "content": "You are a text summarization assistant."},
        {"role": "user", "content": prompt},
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL, 
            messages=messages, 
            max_tokens=1024, 
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Failed to summarize text: {e}")


def generate_post_without_name(topic, target_object, paragraphs):
    """
    Generates a forum post about the topic without mentioning its name.
    
    Args:
        topic (str): The type of topic (movie, celebrity, landmark)
        target_object (str): The name of the object (should not appear in output)
        paragraphs (list): List of paragraph strings with information
        
    Returns:
        str: The generated forum post
    """
    # Get template and system message from templates module
    try:
        template = get_template(topic)
        system_content = get_system_message(topic)
    except ValueError as e:
        raise ValueError(f"Invalid topic '{topic}': {e}")
    
    formatted_paragraphs = "\n".join([f"- {para}" for para in paragraphs])
    prompt = template.format(
        ToTObject=target_object, 
        topic=topic, 
        Psg=formatted_paragraphs
    )
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL, 
            messages=messages, 
            max_tokens=1024, 
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Failed to generate post: {e}")


def generate_single(target_object, topic, output_file_path, wikipedia_url, results_dir_prefix, entity_id=None, query_id=None, paragraphs_file=None):
    """
    Generates a single forum post for a given target object.
    
    Args:
        target_object (str): The name of the object to generate a post about
        topic (str): The type of topic ("movie", "celebrity", "landmark")
        output_file_path (str): Path to save the output TSV file
        wikipedia_url (str, optional): Wikipedia URL for the object
        results_dir_prefix (str): Directory prefix for saving results
        entity_id (str, optional): Entity ID from the TSV file
        query_id (str, optional): Query ID from the TSV file
        paragraphs_file (str, optional): Directory containing JSON files with paragraphs to use instead of Wikipedia
    Returns:
        dict: Results including the generated post and metadata
    """
    # Validate topic
    available_topics = list_available_topics()
    if topic not in available_topics:
        raise ValueError(f"Invalid topic '{topic}'. Available topics: {available_topics}")

    try:
        # Get paragraphs either from file or Wikipedia
        if paragraphs_file:
            print(f"Reading paragraphs from directory: {paragraphs_file}")
            paragraphs = read_paragraphs_from_file(paragraphs_file, target_object)
            print(f"Read {len(paragraphs)} paragraphs from JSON file")
        else:
            # Fetch and process document from Wikipedia
            print(f"Fetching Wikipedia content for: {target_object}")
            doc = fetch_document_from_wiki_full(target_object)
            
            if doc.startswith("Failed") or doc.startswith("No pages"):
                raise Exception(f"Could not fetch Wikipedia content: {doc}")
            
            print("Summarizing content...")
            content_type = get_content_type(topic)
            summarization = summarize_text(doc, content_type)
            paragraphs = split_document(summarization)
        
        if not paragraphs:
            raise Exception("No valid paragraphs found after processing")
        
        print("Generating forum post...")
        response = generate_post_without_name(topic, target_object, paragraphs)
        
        print("-" * 20)
        print(response)
        print("-" * 20)
        
        # Get template for metadata
        template = get_template(topic)
        
        # Prepare result data
        result = {
            "query_id": query_id,
            "entity_id": entity_id,
            "topic": topic,
            "target_object": target_object,
            "paragraphs": paragraphs,
            "template_used": template,
            "response": response,
            "wikipedia_url": wikipedia_url
        }
        
        # Save to JSON file
        json_filename = generate_json_filename(target_object)
        with open(f'{results_dir_prefix}{json_filename}', "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"Saved detailed results to: {json_filename}")
        
        # Extract content from code block or use whole response
        post_content = extract_code_block_content(response, target_object, entity_id, query_id)

        # Validate that target object name doesn't appear in response
        if target_object.lower() in post_content.lower():
            log_warning_to_file(
                "target_name_in_response",
                f"Generated post contains the target object name: {target_object}",
                target_object, entity_id, query_id
            )

        # Process response for TSV output
        processed_response = post_content.replace("\n", " ").strip('"')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Write to TSV file
        file_exists = os.path.isfile(output_file_path)
        
        with open(output_file_path, "a", encoding="utf-8", newline="") as outfile:
            fieldnames = ["QuestionBody", "wikipediaURL", "totObj", "entityId", "queryId"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                "QuestionBody": processed_response,
                "wikipediaURL": wikipedia_url,
                "totObj": target_object,
                "entityId": entity_id or "",
                "queryId": query_id or "",
            })
        
        print(f"Appended result to TSV file: {output_file_path}")
        # print(f"\nGenerated post:\n{response}")
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to process {target_object}: {e}"
        print(f"Error: {error_msg}")
        raise Exception(error_msg)


def main():
    """
    Main function to process entities from TSV file and generate forum posts.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate LLM queries from entity classification TSV")
    parser.add_argument("--start", type=int, default=0, help="Start index for processing entities (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End index for processing entities (default: all entities)")
    parser.add_argument("--use-generic", action="store_true", help="Use 'generic' topic for all entities instead of mapping from category")
    parser.add_argument("--paragraphs-file", type=str, help="Directory containing JSON files with pre-written paragraphs to use instead of fetching from Wikipedia")
    args = parser.parse_args()
    
    # Show available topics
    print("Available topics:", list_available_topics())
    
    # Configuration
    tsv_file_path = "outputs/classification/dev3_first_100_entity_classification.tsv"
    
    # Adjust output paths based on whether generic mode is used
    current_date = datetime.now().strftime("%y%m%d")
    model_path = MODEL.replace('/', '_') + f"_{current_date}"
    if args.use_generic:
        output_file = f"outputs/generated_query/{model_path}_generic/queries.tsv"
        json_dir = f"outputs/generated_query/{model_path}_generic/json_files/"
        warnings_file = f"outputs/generated_query/{model_path}_generic/warnings.tsv"
        failed_file = f"outputs/generated_query/{model_path}_generic/failed_entries.tsv"
    else:
        output_file = f"outputs/generated_query/{model_path}/queries.tsv"
        json_dir = f"outputs/generated_query/{model_path}/json_files/"
        warnings_file = f"outputs/generated_query/{model_path}/warnings.tsv"
        failed_file = f"outputs/generated_query/{model_path}/failed_entries.tsv"
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    try:
        # Read entities from TSV file
        print(f"Reading entities from: {tsv_file_path}")
        entities = read_entities_from_tsv(tsv_file_path, use_generic=args.use_generic)
        print(f"Found {len(entities)} entities to process")
        
        if args.use_generic:
            print("Using 'generic' topic for all entities (ignoring category from CSV)")
        else:
            print("Using category-based topic mapping")
            
        if args.paragraphs_file:
            print(f"Using paragraphs from JSON files in directory: {args.paragraphs_file}")
        else:
            print("Fetching content from Wikipedia and summarizing")
        
        # Apply start and end indices
        start_idx = args.start
        end_idx = args.end if args.end is not None else len(entities)
        
        # Validate indices
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(entities):
            end_idx = len(entities)
        if start_idx >= end_idx:
            print(f"Error: start index ({start_idx}) must be less than end index ({end_idx})")
            return
        
        entities_to_process = entities[start_idx:end_idx]
        print(f"Processing entities from index {start_idx} to {end_idx-1} ({len(entities_to_process)} entities)")
        
        # Process each entity
        successful_count = 0
        failed_count = 0
        
        for i, entity in enumerate(entities_to_process, 1):
            actual_index = start_idx + i - 1
            print(f"\n[{i}/{len(entities_to_process)}] (Index {actual_index}) Processing: {entity['title']} (Category: {entity['category']}, Topic: {entity['topic']})")
            
            try:
                result = generate_single(
                    target_object=entity['title'],
                    topic=entity['topic'],
                    output_file_path=output_file,
                    wikipedia_url=entity['url'],
                    results_dir_prefix=json_dir,
                    entity_id=entity['entity_id'],
                    query_id=entity['query_id'],
                    paragraphs_file=args.paragraphs_file
                )
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                error_message = str(e)
                print(f"✗ Failed to process {entity['title']}: {error_message}")
                
                # Log the failed entry
                log_failed_entry(
                    query_id=entity['query_id'],
                    entity_id=entity['entity_id'],
                    title=entity['title'],
                    error_message=error_message
                )
                continue
        
        print(f"\n=== Processing Complete ===")
        print(f"Processed range: {start_idx} to {end_idx-1}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"Total processed: {len(entities_to_process)}")
        print(f"Output saved to: {output_file}")
        
        # Save warnings to file
        if warning_records:
            save_warnings_to_file(warnings_file)
            print(f"Warnings saved to: {warnings_file}")
            print(f"Total warnings: {len(warning_records)}")
        else:
            print("No warnings recorded.")
        
        # Save failed entries to file
        if failed_records:
            save_failed_entries_to_file(failed_file)
            print(f"Failed entries saved to: {failed_file}")
            print(f"Total failed entries: {len(failed_records)}")
        else:
            print("No failed entries recorded.")
        
    except Exception as e:
        print(f"Failed to process TSV file: {e}")
        # Still save warnings and failed entries if any were collected before the error
        if warning_records:
            save_warnings_to_file(warnings_file)
            print(f"Warnings saved to: {warnings_file}")
        if failed_records:
            save_failed_entries_to_file(failed_file)
            print(f"Failed entries saved to: {failed_file}")


if __name__ == "__main__":
    main()
