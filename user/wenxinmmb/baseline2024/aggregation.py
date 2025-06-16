import json
from glob import glob

# Step 1: Read all JSON files
file_paths = glob("runs/search_result/train/search_results_*.json")
aggregated_data = {}

for file_path in file_paths:
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            query_id = entry["query_id"]
            if query_id not in aggregated_data:
                aggregated_data[query_id] = {"scores": [], "raw_doc_ids": [], "translated_doc_ids": []}
            aggregated_data[query_id]["scores"].extend(entry["scores"])
            aggregated_data[query_id]["raw_doc_ids"].extend(entry["raw_doc_ids"])
            aggregated_data[query_id]["translated_doc_ids"].extend(entry["translated_doc_ids"])

# Step 2: Sort and rearrange data for each query_id
sorted_results = {}
for query_id, data in aggregated_data.items():
    combined = list(zip(data["scores"], data["raw_doc_ids"], data["translated_doc_ids"]))
    combined.sort(reverse=True, key=lambda x: x[0])  # Sort by scores in descending order
    sorted_scores, sorted_raw_doc_ids, sorted_translated_doc_ids = zip(*combined)

    # Process translated_doc_ids to retain only the first number and remove duplicates
    doc_ids = []
    doc_scores = []
    seen_ids = set()
    for translated_id in sorted_translated_doc_ids:
        first_number = translated_id.split("_")[0]
        if first_number not in seen_ids:
            seen_ids.add(first_number)
            doc_ids.append(first_number)
            doc_scores.append(sorted_scores[sorted_translated_doc_ids.index(translated_id)])
        if len(doc_ids) >= 1000:
            break
    
    sorted_results[query_id] = {
        "scores": list(sorted_scores),
        "raw_doc_ids": list(sorted_raw_doc_ids),
        "passage_ids": list(sorted_translated_doc_ids),
        "doc_ids": doc_ids,
        "doc_scores": doc_scores
    }

# Step 3: Write the aggregated and sorted results to a new JSON file
with open("runs/search_result/train/aggregated_results.json", "w") as output_file:
    for query_id, result in sorted_results.items():
        output_file.write(json.dumps({"query_id": query_id, **result}) + "\n")

# Step 4: Generate the qrel file
with open("runs/search_result/train/bge_2025_train.txt", "w") as run_file:
    for query_id, result in sorted_results.items():
        for i, doc_id in enumerate(result["doc_ids"]):
            score = result["doc_scores"][i]
            run_file.write(f"{query_id}\tQ0\t{doc_id}\t{i}\t{score}\tbge-m3\n")