from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="Upstash/wikipedia-2024-06-bge-m3", repo_type='dataset',
#                 filename="data/en/*.parquet",
#                 local_dir="/Users/wenxin/tot/tot_data/upstash-embed")

from huggingface_hub import snapshot_download
snapshot_download(repo_id="Upstash/wikipedia-2024-06-bge-m3", repo_type='dataset',
                   local_dir="/Users/wenxin/tot/tot_data/upstash-embed",
                   allow_patterns=["data/en/*.parquet"])