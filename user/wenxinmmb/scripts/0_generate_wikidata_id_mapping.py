import json
from wikimapper import WikiMapper
from tqdm import tqdm
# Install and setup wikimapper according to the instructions at:
# https://github.com/jcklie/wikimapper/tree/master

# Paths (update if needed)

project_root = "/home/wenxin/project"
CORPUS_PATH = f"{project_root}/data/2025/corpus.jsonl"
WIKIMAPPER_DB_PATH = f"{project_root}/data/wikimapper-dump/index_enwiki-20250620.db"
OUTPUT_PATH = f"{project_root}/data/id2wikidataid_6m.json"

# Load wikimapper
mapper = WikiMapper(WIKIMAPPER_DB_PATH)

mapping = {}

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Generating mapping", unit="line"):
        obj = json.loads(line)
        wiki_id = str(obj["id"])
        wikidata_id = mapper.wikipedia_id_to_id(wiki_id)
        mapping[wiki_id] = wikidata_id

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f"Mapping written to {OUTPUT_PATH}")
