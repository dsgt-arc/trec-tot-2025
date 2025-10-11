import argparse
import json
import os
from sentence_transformers import SentenceTransformer
import polars as pl

def main():
    parser = argparse.ArgumentParser(description="Embed queries using BGE-M3 model")
    parser.add_argument("--queries_file", required=True, help="Path to queries.jsonl file")
    parser.add_argument("--device", default="cuda", help="Device for embedding (cuda or cpu)")
    args = parser.parse_args()

    # Load queries
    queries = []
    query_ids = []
    with open(args.queries_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            queries.append(obj["query"])
            query_ids.append(obj["query_id"])

    # Load model
    transformer = SentenceTransformer(
        "BAAI/bge-m3",
        device=args.device,
        revision="babcf60cae0a1f438d7ade582983d4ba462303c2",
    )

    # Embed queries
    embeddings = transformer.encode(
        queries,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Save to parquet
    out_dir = os.path.dirname(args.queries_file)
    out_path = os.path.join(out_dir, "query-bge-embed.parquet")
    df = pl.DataFrame({
        "query_id": query_ids,
        "embedding": embeddings.tolist()
    })
    df.write_parquet(out_path)
    print(f"Saved embeddings to {out_path}")

if __name__ == "__main__":
    main()

