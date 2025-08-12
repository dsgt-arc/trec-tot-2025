import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os
prefix = "/home/wenxin/project/data/2025/generated-queries/llm-set1"

def split_qrel_and_queries():
    # Create necessary directories if they don't exist
    os.makedirs(prefix, exist_ok=True)
    os.makedirs(os.path.join(prefix, "train"), exist_ok=True)
    os.makedirs(os.path.join(prefix, "dev"), exist_ok=True)
    os.makedirs(os.path.join(prefix, "test"), exist_ok=True)
    
    # File paths - adjust these paths as needed
    qrel_path = os.path.join(prefix, "qrel.txt")
    queries_path = os.path.join(prefix, "queries.jsonl")
    
    # Check if files exist
    if not os.path.exists(qrel_path):
        print(f"Error: {qrel_path} not found")
        return
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found")
        return
    
    # Read qrel file
    print("Reading qrel.txt...")
    qrel_df = pd.read_csv(qrel_path, sep='\s+', names=['query_id', 'iter', 'doc_id', 'relevance'])
    
    # Read queries jsonl file
    print("Reading queries.jsonl...")
    queries = []
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    
    queries_df = pd.DataFrame(queries)
    
    # Split queries data based on query IDs (assuming 'query_id' field in queries)
    # Adjust the field name if different (e.g., 'query_id', '_id', etc.)
    query_id_field = 'query_id'  # Change this if your query ID field has a different name
    
    if query_id_field not in queries_df.columns:
        print(f"Warning: '{query_id_field}' field not found in queries. Available fields: {list(queries_df.columns)}")
        print("Please update the query_id_field variable in the script")
        return
    
    # Get unique query IDs from qrel
    unique_query_ids = qrel_df['query_id'].unique()
    print(f"Total unique queries: {len(unique_query_ids)}")
    
    # Debug: Check data types and sample values
    print(f"Sample qrel query_ids: {list(unique_query_ids[:5])}")
    print(f"Sample queries query_ids: {list(queries_df[query_id_field].head())}")
    print(f"qrel query_id type: {type(unique_query_ids[0])}")
    print(f"queries query_id type: {type(queries_df[query_id_field].iloc[0])}")
    
    # Convert all query IDs to string for consistency
    print("Converting all query IDs to string type...")
    unique_query_ids_str = [str(qid) for qid in unique_query_ids]
    queries_df[query_id_field] = queries_df[query_id_field].astype(str)
    
    # Split query IDs: 75% train, 15% dev, 10% test
    train_queries, temp_queries = train_test_split(
        unique_query_ids_str, 
        test_size=0.25,  # 25% for dev + test
        random_state=42
    )
    
    dev_queries, test_queries = train_test_split(
        temp_queries,
        test_size=0.4,  # 40% of 25% = 10% of total (test), remaining 60% of 25% = 15% of total (dev)
        random_state=42
    )
    
    print(f"Train queries: {len(train_queries)}")
    print(f"Dev queries: {len(dev_queries)}")
    print(f"Test queries: {len(test_queries)}")
    
    # Convert back to original type for qrel filtering
    train_queries_orig = [int(qid) for qid in train_queries]
    dev_queries_orig = [int(qid) for qid in dev_queries]
    test_queries_orig = [int(qid) for qid in test_queries]
    
    # Split qrel data based on query IDs
    train_qrel = qrel_df[qrel_df['query_id'].isin(train_queries_orig)]
    dev_qrel = qrel_df[qrel_df['query_id'].isin(dev_queries_orig)]
    test_qrel = qrel_df[qrel_df['query_id'].isin(test_queries_orig)]
    
    print(f"Train qrel entries: {len(train_qrel)}")
    print(f"Dev qrel entries: {len(dev_qrel)}")
    print(f"Test qrel entries: {len(test_qrel)}")
    
    # Split queries data based on query IDs (now both are strings)
    train_queries_df = queries_df[queries_df[query_id_field].isin(train_queries)]
    dev_queries_df = queries_df[queries_df[query_id_field].isin(dev_queries)]
    test_queries_df = queries_df[queries_df[query_id_field].isin(test_queries)]
    
    print(f"Train queries entries: {len(train_queries_df)}")
    print(f"Dev queries entries: {len(dev_queries_df)}")
    print(f"Test queries entries: {len(test_queries_df)}")
    
    # Save qrel splits
    train_qrel.to_csv(f'{prefix}/train/qrel.txt', sep=' ', index=False, header=False)
    dev_qrel.to_csv(f'{prefix}/dev/qrel.txt', sep=' ', index=False, header=False)
    test_qrel.to_csv(f'{prefix}/test/qrel.txt', sep=' ', index=False, header=False)

    # Save queries splits
    with open(f'{prefix}/train/queries.jsonl', 'w', encoding='utf-8') as f:
        for _, row in train_queries_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')

    with open(f'{prefix}/dev/queries.jsonl', 'w', encoding='utf-8') as f:
        for _, row in dev_queries_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')

    with open(f'{prefix}/test/queries.jsonl', 'w', encoding='utf-8') as f:
        for _, row in test_queries_df.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    
    print("\nSplit completed! Files created:")
    print("- train_qrel.txt, dev_qrel.txt, test_qrel.txt")
    print("- train_queries.jsonl, dev_queries.jsonl, test_queries.jsonl")

if __name__ == "__main__":
    split_qrel_and_queries()