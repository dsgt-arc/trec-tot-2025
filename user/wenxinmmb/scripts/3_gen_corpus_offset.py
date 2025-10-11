import json
import os
from tqdm import tqdm

def generate_corpus_offset():
    """
    Generate a JSON object with Wikipedia ID as key and offset information as values.
    Reads from data/corpus-first-1000.jsonl and saves offsets for each line.
    """
    project_root = "/home/wenxin/project"
    # Define the input file path
    input_file = f"{project_root}/data/2025/corpus.jsonl"
    output_file = f"{project_root}/data/2025/corpus-offset-mapping.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    offset_mapping = {}
    
    print(f"Reading {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Create progress bar without total count
        progress_bar = tqdm(desc="Processing lines", unit="lines")
        
        while True:
            # Get the current position before reading the line
            line_start = f.tell()
            line = f.readline()
            
            # If we've reached the end of file, break
            if not line:
                break
            
            # Update progress bar
            progress_bar.update(1)
            
            # Get the position after reading the line (end of current line)
            line_end = f.tell()
            
            try:
                # Parse the JSON line
                data = json.loads(line.strip())
                
                # Extract the Wikipedia ID
                wiki_id = data.get('id')
                
                if wiki_id:
                    # Store the offset information
                    offset_mapping[wiki_id] = {
                        "offset_start": line_start,
                        "offset_end": line_end - 1  # Subtract 1 to exclude the newline character
                    }
                else:
                    progress_bar.write(f"Warning: No 'id' field found in line starting at offset {line_start}")
                    
            except json.JSONDecodeError as e:
                progress_bar.write(f"Warning: Failed to parse JSON at offset {line_start}: {e}")
                continue
        
        # Close the progress bar
        progress_bar.close()
    
    # Save the offset mapping to a JSON file
    print(f"Saving offset mapping to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # json.dump(offset_mapping, f, indent=2, ensure_ascii=False)
        json.dump(offset_mapping, f, ensure_ascii=False)
    
    print(f"Successfully generated offset mapping for {len(offset_mapping)} entries")
    print(f"Output saved to: {output_file}")
    
    # Print first few entries as example
    print("\nFirst 5 entries in the mapping:")
    for i, (wiki_id, offsets) in enumerate(offset_mapping.items()):
        if i >= 5:
            break
        print(f"  ID: {wiki_id} -> Start: {offsets['offset_start']}, End: {offsets['offset_end']}")

if __name__ == "__main__":
    generate_corpus_offset()