import argparse
import json
import logging
import os
import re
import subprocess
from collections import Counter, defaultdict

import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from thefuzz import fuzz, process
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils

log = logging.getLogger("name_match_score")

# Command:
# python name_match_score.py --input output/dev3-o4-mini.jsonl --split dev3 --index_name llm_title_alias --run output/dev3-o4-mini-70-25.run --run_id o4-mini_alias

def get_llm_response_titles(query):
    # Extract entity names from LLM response format with title and relevance_score/score
    assert 'result' in query, "query should have 'result' field"
    if 'error' in query['result']:
        return []  # skip queries with errors

    result = query['result']
    
    # Handle format: entities is a list of dicts with 'title' and 'relevance_score'/'score'
    if 'entities' in result and isinstance(result['entities'], list):
        entities = result['entities']
        if entities and isinstance(entities[0], dict):
            # Check if entities have title and score fields
            if 'title' in entities[0]:
                # Sort by relevance_score or score in descending order
                score_key = 'relevance_score' if 'relevance_score' in entities[0] else 'score'
                if score_key in entities[0]:
                    # Sort by score descending, then extract titles
                    sorted_entities = sorted(entities, key=lambda x: x.get(score_key, 0), reverse=True)
                    return [entity['title'] for entity in sorted_entities]
                else:
                    # No score field, just return titles in original order
                    return [entity['title'] for entity in entities]
    
    # If format doesn't match expected structure, return empty list
    print(f"[WARN] Could not extract entity names for query: {query.get('query_id')}, result: {result}")
    return []

def create_title_index(dataset, dest_folder, index):
    log.info(f"creating files for indexing in {dest_folder}")
    docs_folder = os.path.join(dest_folder, "docs")
    os.makedirs(docs_folder, exist_ok=True)

    # get aliases - always gather wikidata aliases
    aliases = {}   

    # load the alias json file
    alias_path = "/home/wenxin/project/data/id_to_aliases.json"
    if not os.path.exists(alias_path):
        raise ValueError(f"alias file not found: {alias_path}. Double check the path or run the script to generate it.")
    log.info(f"loading aliases from {alias_path}")
    aliases = json.load(open(alias_path, "r", encoding="utf-8"))

    with open(os.path.join(docs_folder, "docs.jsonl"), "w") as writer:
        for raw_doc in tqdm(dataset.docs_iter(), desc="gathering aliases"):
            # Using 2025 data format
            doc_id = raw_doc.id
            # Always use wikidata aliases
            original_list = aliases[doc_id].copy()
            aliases[doc_id] = set(original_list)

            # remove braces and add to aliases
            no_br = set()
            for _ in aliases[doc_id]:
                no_br.add(remove_braces(_))
            aliases[doc_id].update(no_br)

            doc = {
                "id": doc_id,
                "contents": "\n".join(aliases[doc_id])
            }
            writer.write(json.dumps(doc) + "\n")

    # call pyserini indexer
    cmd = f"""python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input {docs_folder} \
      --index {index} \
      --generator DefaultLuceneDocumentGenerator \
      --keepStopwords \
      --stemmer none \
      --threads 1 \
      --storeRaw""".split()

    try:
        subprocess.call(cmd)
    except subprocess.CalledProcessError as e:
        log.exception("Exception occurred during indexing!")
        raise ValueError(e)

    return aliases

def remove_braces(text):
    return re.sub("[\(].*?[\)]", "", text).strip()

def remove_non_alpha(text):
    return re.sub(r'[\W\s]', ' ', text)

def resolve(title, matched_title, title_to_doc_id, aliases, scorer, assert_perfect_score=False):
    gen = []
    for doc_id in title_to_doc_id[matched_title]:
        # pick the best match
        best_match, score = process.extractOne(title, aliases[doc_id], scorer=scorer)

        if assert_perfect_score:
            # perfect match, this *has* to happen
            assert score == 100

        if score == 100:
            score = 101
        gen.append((best_match, doc_id, score))

    return gen

if __name__ == '__main__':

    parser = argparse.ArgumentParser("name_match_score", description="post process output from LLM, and compute run")
    parser.add_argument("--input", required=True, help="output from LLM (json)")

    parser.add_argument("--split", required=True, help="corresponding split i.e 'train', 'dev1', 'dev2','dev3', 'test'")
    parser.add_argument("--index_name", required=True, help="name of index")
    parser.add_argument("--run", required=True, help="path to save run")
    parser.add_argument("--run_format", default=None, choices={"trec_eval"})
    parser.add_argument("--run_id", required=True, help="run id (required if run_format = trec_eval)")
    parser.add_argument("--ref_run", default=None, help="if provided, this run is used to break ties")
    parser.add_argument("--docs_path", default="./anserini_title_docs",
                        help="path to store (temp) documents for indexing")
    parser.add_argument("--index_path", default="./anserini_title_indices", help="path to store (all) indices")
    parser.add_argument("--n_threads", default=8, type=int, help="number of threads (eval)")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size (eval) ")
    parser.add_argument("--min_score", default=70, type=int, help="minimum fuzzy matching score threshold")
    parser.add_argument("--bm25_k", default=10, type=int, help="number of BM25 results to retrieve")

    logging.basicConfig(level=logging.INFO)
    log.setLevel(logging.INFO)

    args = parser.parse_args()
    docs_path = os.path.join(args.docs_path, args.index_name)
    index = os.path.join(args.index_path, args.index_name)
    # Always use wikidata aliases
    AL_PATH = "/home/wenxin/project/data/id_to_aliases.json"
    if not os.path.exists(index):
        log.warning(f"Index {index} does not exist, creating index with script llm_match_name.py!")
        assert False, "Indexing is not supported in this script. Please run llm_match_name.py to create the index."
    else:
        aliases = utils.read_json(AL_PATH)
        log.info("index already created. loaded aliases")

    # title -> set{doc_id}
    title_to_doc_id = {}
    for doc_id, titles in tqdm(aliases.items(), leave=False):
        for al in aliases[doc_id]:
            if al in title_to_doc_id:
                title_to_doc_id[al].update([doc_id])
            else:
                title_to_doc_id[al] = {doc_id}

    queries = utils.read_jsonl(args.input)

    searcher = LuceneSearcher(index)

    titles = []
    for query in queries:
        title_list = get_llm_response_titles(query)
        print(f"query: {query['query_id']}, titles: {title_list}")
        titles.extend(title_list)

    # dedup
    titles = list(set(titles))
    log.info(f"performing search on title index for {len(titles)} titles")

    # title -> [(title, doc_id, score))
    # score == 101 if it's a perfect match
    gen_title_to_doc_ids = {}
    matches = Counter()
    unmatched = set()
    unmatched_props = {}
    MIN_SCORE = args.min_score
    BM25_K = args.bm25_k

    scorer = fuzz.ratio

    # The 4 mapping strategies
    # First: Exact title match
    # Second: Fuzzy match against BM25 retrieved choices
    # Third: Remove brackets/parentheses from choices and match
    # Fourth (new): Remove brackets/parentheses from the original title and match against original choices
    for title in tqdm(titles):
        if title in title_to_doc_id:
            gen_title_to_doc_ids[title] = resolve(title=title,
                                                  matched_title=title,
                                                  title_to_doc_id=title_to_doc_id,
                                                  aliases=aliases,
                                                  scorer=scorer,
                                                  assert_perfect_score=True)

            if len(gen_title_to_doc_ids[title]) == 1:
                matches["exact_1"] += 1
            else:
                matches["exact_n"] += 1
            print('title:', title, 'strategy 1 - exact matches:', gen_title_to_doc_ids[title])
        else:

            # no exact match, perform retrieval, followed by matching
            res = searcher.search(title, k=BM25_K)

            choices = []
            for doc in res:
                # get the closest alias
                docid = doc.docid
                best_match, score = process.extractOne(title, aliases[docid])
                choices.append(best_match)

            # Try strategy 2: Direct fuzzy matching against choices
            matched = process.extractOne(title, choices)
            if matched is not None:
                matched_title, score = matched
                if score >= MIN_SCORE:
                    gen_title_to_doc_ids[title] = resolve(title=title,
                                                          matched_title=matched_title,
                                                          title_to_doc_id=title_to_doc_id,
                                                          aliases=aliases,
                                                          scorer=scorer,
                                                          assert_perfect_score=False)

                    if len(gen_title_to_doc_ids[title]) == 1:
                        matches["strategy2_1"] += 1
                        print(title, gen_title_to_doc_ids[title])
                    else:
                        matches["strategy2_n"] += 1
                    print('title:', title, 'strategy 2 - fuzzy matches against BM25 retrieved choices:', gen_title_to_doc_ids[title])

            # Try strategy 3: Remove braces and non-alpha from both title and choices
            if title not in gen_title_to_doc_ids:
                # we need to retain the original titles for mapping it back
                nobr2br = {remove_non_alpha(remove_braces(_)): _ for _ in choices}
                choices_nobr = list(nobr2br.keys())
                matched_nobr = process.extractOne(remove_non_alpha(remove_braces(title)), choices_nobr)
                if matched_nobr is not None:
                    matched_title_nobr, score_nobr = matched_nobr
                    if score_nobr >= MIN_SCORE:
                        matched_org_title = nobr2br[matched_title_nobr]

                        gen_title_to_doc_ids[title] = resolve(title=title,
                                                              matched_title=matched_org_title,
                                                              title_to_doc_id=title_to_doc_id,
                                                              aliases=aliases,
                                                              scorer=scorer,
                                                              assert_perfect_score=False)

                        if len(gen_title_to_doc_ids[title]) == 1:
                            matches["strategy3_1"] += 1
                        else:
                            matches["strategy3_n"] += 1
                        print('title:', title, 'strategy 3 - fuzzy matches after removing brackets/non-alpha from both title and choices:', gen_title_to_doc_ids[title])

            # Try strategy 4: Remove brackets and non-alpha from original title, do BM25 search, then match
            if title not in gen_title_to_doc_ids:
                title_normalized = remove_non_alpha(remove_braces(title)).strip()
                if title_normalized != title:  # Only proceed if normalization changed the title
                    # Perform BM25 search with normalized title
                    res_normalized = searcher.search(title_normalized, k=BM25_K)
                    
                    choices_normalized = []
                    for doc in res_normalized:
                        # get the closest alias
                        docid = doc.docid
                        best_match, score = process.extractOne(title_normalized, aliases[docid])
                        choices_normalized.append(best_match)
                    
                    # Create mapping from normalized choices back to original choices
                    norm2orig = {remove_non_alpha(remove_braces(_)).strip(): _ for _ in choices_normalized}
                    choices_norm_list = list(norm2orig.keys())
                    
                    # Match normalized title against normalized choices
                    matched_norm = process.extractOne(title_normalized, choices_norm_list)
                    if matched_norm is not None:
                        matched_title_norm, score_norm = matched_norm
                        if score_norm >= MIN_SCORE:
                            matched_original_title = norm2orig[matched_title_norm]
                            
                            gen_title_to_doc_ids[title] = resolve(title=title,
                                                                  matched_title=matched_original_title,
                                                                  title_to_doc_id=title_to_doc_id,
                                                                  aliases=aliases,
                                                                  scorer=scorer,
                                                                  assert_perfect_score=False)

                            if len(gen_title_to_doc_ids[title]) == 1:
                                matches["strategy4_1"] += 1
                            else:
                                matches["strategy4_n"] += 1
                            print('title:', title, 'strategy 4 - fuzzy matches after normalizing title and doing new BM25 search:', gen_title_to_doc_ids[title])

            # If no strategy worked, add to unmatched
            if title not in gen_title_to_doc_ids:
                unmatched.add(title)
                unmatched_props[title] = {
                    "choices": choices,
                    "matched": matched if 'matched' in locals() else None,
                    "matched_nobr": matched_nobr if 'matched_nobr' in locals() else None,
                    "title_normalized": title_normalized if 'title_normalized' in locals() else None,
                    "matched_norm": matched_norm if 'matched_norm' in locals() else None
                }
                print('title:', title, 'no match found after all strategies')

    print(matches)
    print(f"unmatched: {len(unmatched)}")
    rows = []

    for title in unmatched:
        row = {
            "title": title,
            "choices": ";".join(unmatched_props[title]["choices"]),
            "matched": unmatched_props[title].get("matched"),
            "matched_nobr": unmatched_props[title].get("matched_nobr"),
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(f"unmatched/unmatched_{args.split}.csv", index=False)

    if args.ref_run:
        log.info(f"using reference run: {args.ref_run}")
        ref_run = defaultdict(dict)
        with open(args.ref_run) as reader:
            for line in reader:
                qid, _, doc_id, _, score, _ = line.split()
                ref_run[qid][doc_id] = float(score)
    else:
        log.info("no reference run provided!")
        ref_run = None

    # qid -> doc_id -> relevance
    run = {}
    # create run
    for query in queries:
        qid = query["query_id"]
        run[qid] = {}
        llm_titles = get_llm_response_titles(query)
        ranks = range(len(llm_titles), 0, -1)
        for rank, title in zip(ranks, llm_titles):
            gen_titles = gen_title_to_doc_ids.get(title, [])
            # no matches! :(
            if len(gen_titles) == 0:
                continue
            # single match, no problem!
            elif len(gen_titles) == 1:
                matched_title, doc_id, score = gen_titles[0]
                run[qid][doc_id] = float(rank)
            # if reference run isn't provided, assign the same
            # rank to each matched title
            elif ref_run is None:
                # assign same rank
                for (matched_title, doc_id, score) in gen_titles:
                    run[qid][doc_id] = float(rank)
            # otherwise re-order based on reference run
            else:
                ref_scores = {}
                rank = float(rank)
                for (matched_title, doc_id, score) in gen_titles:
                    if doc_id in ref_run[qid]:
                        ref_scores[doc_id] = ref_run[qid][doc_id]
                    else:
                        # those without ref scores get score = rank
                        run[qid][doc_id] = rank

                # those with ref scores gets score from (rank+step to rank+step*len(ref_scores))
                step = 1 / (len(ref_scores) + 1)
                for srank, (doc_id, _) in enumerate(sorted(ref_scores.items(), key=lambda _: _[1])):
                    rank += step
                    run[qid][doc_id] = rank

    # write run file
    run_id = args.run_id
    with open(args.run, "w") as writer:
        for qid, r in run.items():
            for rank, (doc_id, score) in enumerate(sorted(r.items(), key=lambda _: -_[1])):
                writer.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{float(score)}\t{run_id}\n")
