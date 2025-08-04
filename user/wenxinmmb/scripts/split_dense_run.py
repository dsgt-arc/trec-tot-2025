import os

# Define input file and output mappings
input_file = "outputs/dev3-gen-all-dense-run.txt"
output_folder = "outputs"
output_files = {
    "1": "dev3-g1-dense-run.txt",
    "2": "dev3-gpt4o-dense-run.txt", 
    "3": "dev3-o4-mini-dense-run.txt",
    "4": "dev3-random-dense-run.txt",
    "5": "dev3-wikitext-dense-run.txt"
}

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize output file handles
output_handles = {}
for prefix, filename in output_files.items():
    output_path = os.path.join(output_folder, filename)
    output_handles[prefix] = open(output_path, "w")

try:
    # Read input file and split lines based on query ID prefix
    with open(input_file, "r") as infile:
        for line in infile:
            # Extract the first character of the query ID (should be the prefix)
            parts = line.strip().split()
            if parts:
                query_id = parts[0]
                prefix = query_id[0]  # First character is the prefix
                
                # Write to appropriate output file if prefix matches
                if prefix in output_handles:
                    # Remove the prefix from query ID and reconstruct the line
                    new_query_id = query_id[1:]  # Remove first character (prefix)
                    parts[0] = new_query_id
                    new_line = ' '.join(parts) + '\n'
                    output_handles[prefix].write(new_line)
                else:
                    print(f"Warning: Unknown prefix '{prefix}' in query ID '{query_id}'")

finally:
    # Close all output files
    for handle in output_handles.values():
        handle.close()

print("File splitting completed!")
print(f"Input file: {input_file}")
print(f"Output folder: {output_folder}")
for prefix, filename in output_files.items():
    output_path = os.path.join(output_folder, filename)
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            line_count = sum(1 for _ in f)
        print(f"  {filename}: {line_count} lines")
