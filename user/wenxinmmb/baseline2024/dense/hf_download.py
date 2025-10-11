from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser("hf_download",
                        description="download huggingface data")

parser.add_argument("--local_dir", required=True, type=str,
                    help="Directory to store the downloaded data, e.g., /Users/wenxin/tot/tot_data/upstash-embed")

args = parser.parse_args()

snapshot_download(repo_id="Upstash/wikipedia-2024-06-bge-m3", repo_type='dataset',
                   local_dir=args.local_dir,
                   allow_patterns=["data/en/*.parquet"])
