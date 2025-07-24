import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch

# Load the classifier
MODEL_NAME = "davanstrien/ModernBERT-web-topics-1m"
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device, truncation=True)

def extract_first_paragraph(text: str) -> str:
    """Extracts the first paragraph from the full Wikipedia article text."""
    if not isinstance(text, str):
        return ""
    # Split on double newlines or newline+space
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    return parts[0] if parts else text[:500]  # fallback to first 500 chars

def classify_wikipedia_entries(df: pd.DataFrame, batch_size = 1024):
    results = []
    input_texts = []

    # Preprocess: Build inputs like "Title - First Paragraph"
    for _, row in df.iterrows():
        first_paragraph = extract_first_paragraph(row['text'])
        input_texts.append(f"{row['title'].strip()} - {first_paragraph}")

    # Batched inference
    all_predictions = []
    for i in tqdm(range(0, len(input_texts), batch_size), desc="Running classification"):
        batch = input_texts[i:i+batch_size]
        preds = classifier(batch, truncation=True, top_k=1)
        all_predictions.extend(preds)

    # Zip results back to doc_ids and titles
    for idx, pred in enumerate(all_predictions):
        results.append({
            "id": df.iloc[idx]["id"],
            "title": df.iloc[idx]["title"],
            "predicted_topic": pred[0]["label"],
            "confidence": round(pred[0]["score"], 4)
        })

    return pd.DataFrame(results)


def process_parquet_shards(shard_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(shard_dir)):
        if not fname.endswith(".parquet"):
            continue
        
        # skip if already processed
        if os.path.exists(os.path.join(output_dir, fname.replace(".parquet", "_topics.csv"))):
            print(f"Skipping {fname} - already processed")
            continue
        
        in_path = os.path.join(shard_dir, fname)
        out_path = os.path.join(output_dir, fname.replace(".parquet", "_topics.csv"))

        print(f"Processing: {fname}")
        df = pd.read_parquet(in_path)

        # Ensure required columns
        if not {'id', 'title', 'text'}.issubset(df.columns):
            print(f"Skipping {fname} - missing one of required columns: id, title, text")
            continue

        classified_df = classify_wikipedia_entries(df)
        classified_df.to_csv(out_path, index=False)
        print(f"Saved to: {out_path}")

# Run it
if __name__ == "__main__":
    shard_folder = "/workspace/split_parquet_shards"
    output_folder = "/workspace/classified_topics_output"
    process_parquet_shards(shard_folder, output_folder)
