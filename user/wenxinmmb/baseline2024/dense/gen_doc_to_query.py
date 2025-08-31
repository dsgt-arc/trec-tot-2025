import os

DATA_PATH = os.environ.get("DATA_PATH", "/path/to/data")

def parse_trec_files(file_list):
    doc_query_pairs = []
    for file_path in file_list:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                query_id = parts[0]
                doc_id = parts[2]
                doc_query_pairs.append((int(doc_id), int(query_id)))
    return doc_query_pairs

def main():
    input_files = []
    for split in ['train', 'dev1', 'dev2', 'dev3']:
        input_files.append(f"{DATA_PATH}/results/{split}/bm25.txt")
        input_files.append(f"{DATA_PATH}/2025/{split}-2025/qrel.txt")
    input_files.append(f"{DATA_PATH}/results/2025-test/pt-bm25.txt")

    for split in ['train','dev','test']:
        input_files.append(f"{DATA_PATH}/results/llmset1-{split}/bm25.txt")
        input_files.append(f"{DATA_PATH}/2025/llmset1-{split}-2025/qrel.txt")

    output_file = "outputs/doc_to_query/all-sets.tsv"
    # make sure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    pairs = list(set(parse_trec_files(input_files)))
    pairs.sort()  # sort by doc_id (as integer)
    with open(output_file, 'w') as out:
        for doc_id, query_id in pairs:
            out.write(f"{doc_id}\t{query_id}\n")

if __name__ == "__main__":
    main()