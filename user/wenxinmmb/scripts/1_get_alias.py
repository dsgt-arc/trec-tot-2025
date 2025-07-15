import json

# NOTE: follow instrcution to setup environment -- https://github.com/phucty/wikidb
# With additional steps:
# conda create -n wikidb_new python=3.9 # Use 3.9 instead of 3.6 as indicated in the repo
# conda activate wikidb_new

# conda install -c conda-forge marisa-trie
# conda install -c conda-forge ujson
# conda install -c anaconda psutil
# conda install -c anaconda lz4

import sys
sys.path.append("/home/wenxin/project/data/wikidb") # Paths (edit as needed)
from core.db_wd import DBWikidata
from tqdm import tqdm

# Paths (edit as needed)
project_root = "/home/wenxin/project"
corpus_path = f"{project_root}/data/2025/corpus.jsonl"
id2wikidata_path = f"{project_root}/data/id2wikidataid_6m.json"
output_path = f"{project_root}/data/id_to_aliases.json"

# Load wiki-id to wikidata-id mapping
with open(id2wikidata_path, "r", encoding="utf-8") as f:
    wikiid2wdid = json.load(f)

# Initialize DBWikidata
db = DBWikidata()

result = {}

# Read corpus and build mapping
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing corpus", unit="line"):
        item = json.loads(line)
        wiki_id = str(item["id"])
        title = item.get("title", "")
        wdid = wikiid2wdid.get(wiki_id)
        if wdid:
            aliases = db.get_aliases(wdid, "en")
        else:
            aliases = []
        if aliases:
            result[wiki_id] = aliases + [title]
        else:
            result[wiki_id] = [title]

# Write output
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)