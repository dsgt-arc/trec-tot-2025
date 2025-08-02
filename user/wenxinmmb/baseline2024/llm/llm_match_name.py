import argparse
import json
import logging
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime

import ir_datasets
import pandas as pd
import pytrec_eval
from pyserini.search.lucene import LuceneSearcher
from thefuzz import fuzz, process
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils
from config import global_config

log = logging.getLogger("llm_match_name")

# Command:
# python llm_match_name.py --input output/dev3-o4-mini.jsonl --split dev3 --data_path $DATA_PATH --index_name llm_title_alias --run output/dev3-o4-mini-70-25.run --run_id o4-mini_alias --gather_wikidata_aliases

def get_llm_response_titles(query):
    # Robustly extract entity names from various LLM response formats
    assert 'result' in query, "query should have 'result' field"
    if 'error' in query['result']:
        return []  # skip queries with errors

    result = query['result']
    # Possible keys that may contain entity names
    candidate_keys = [
        'entities',
        'possible_entities',
        'possible_palaces',
        'entity_names',
    ]

    inner_keys = [
        'entity_name',
        'name',
        # 'title'
    ]

    for key in candidate_keys:
        if key in result:
            value = result[key]
            # If it's a list of dicts with 'entity_name', extract those
            if isinstance(value, list) and value and isinstance(value[0], dict):
                # Try 'entity_name' or 'name' as key
                for inner_key in inner_keys:
                    if inner_key in value[0]:
                        return [item[inner_key] for item in value if inner_key in item]
            # If it's a list of strings
            if isinstance(value, list) and (not value or isinstance(value[0], str)):
                return value
    # If 'entity_names' is present (sometimes as a secondary key)
    if 'entity_names' in result:
        return result['entity_names']
    # If none of the above, try to extract the first list of strings or dicts with 'entity_name' or 'name'
    for v in result.values():
        if isinstance(v, list):
            if v and isinstance(v[0], dict):
                for inner_key in inner_keys:
                    if inner_key in v[0]:
                        return [item[inner_key] for item in v if inner_key in item]
            if v and isinstance(v[0], str):
                return v
    # Fallback: return empty list
    print(f"[WARN] Could not extract entity names for query: {query.get('query_id')}, result: {result}")
    return []

def create_title_index(dataset, dest_folder, index, gather_wikidata_aliases):
    log.info(f"creating files for indexing in {dest_folder}")
    docs_folder = os.path.join(dest_folder, "docs")
    os.makedirs(docs_folder, exist_ok=True)

    # get aliases
    aliases = {}   

    if gather_wikidata_aliases:
        # load the alias json file
        alias_path = "/home/wenxin/project/data/id_to_aliases.json"
        if not os.path.exists(alias_path):
            raise ValueError(f"alias file not found: {alias_path}. Double check the path or run the script to generate it.")
        log.info(f"loading aliases from {alias_path}")
        aliases = json.load(open(alias_path, "r", encoding="utf-8"))

    log.info(f"gather_wikidata_aliases: {gather_wikidata_aliases}")
    with open(os.path.join(docs_folder, "docs.jsonl"), "w") as writer:
        for raw_doc in tqdm(dataset.docs_iter(), desc="gathering aliases"):
            if hasattr(raw_doc, 'doc_id'): # for 2024 data
                doc_id = raw_doc.doc_id
            else:  # for 2025 data
                doc_id = raw_doc.id
            if not gather_wikidata_aliases:
                aliases[doc_id] = {raw_doc.title}
            else:
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

    parser = argparse.ArgumentParser("llm_match_name", description="post process output from LLM, and compute run")
    parser.add_argument("--input", required=True, help="output from LLM (json)")

    parser.add_argument("--split", required=True, help="corresponding split i.e 'train', 'dev1', 'dev2','dev3', 'test'")
    parser.add_argument("--gather_wikidata_aliases", action="store_true", default=False,
                        help="if set, gathers aliases from Wikidata (recommended, takes time)")
    parser.add_argument("--data_path", required=True, help="location to dataset")
    parser.add_argument("--index_name", required=True, help="name of index")
    parser.add_argument("--run", required=True, help="path to save run")
    parser.add_argument("--run_format", default=None, choices={"trec_eval"})
    parser.add_argument("--run_id", required=True, help="run id (required if run_format = trec_eval)")
    parser.add_argument("--ref_run", default=None, help="if provided, this run is used to break ties")

    parser.add_argument("--metrics", required=False, default=global_config.METRICS,
                        help="csv - metrics to evaluate")
    parser.add_argument("--docs_path", default="./anserini_title_docs",
                        help="path to store (temp) documents for indexing")
    parser.add_argument("--index_path", default="./anserini_title_indices", help="path to store (all) indices")
    parser.add_argument("--param_k1", default=0.8, type=float, help="param: k1 for BM25")
    parser.add_argument("--param_b", default=1.0, type=float, help="param: b for BM25")
    parser.add_argument("--n_threads", default=8, type=int, help="number of threads (eval)")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size (eval) ")
    parser.add_argument("--data_version", default="2025",
                        help="data version to use, e.g., 2024 or 2025")
    parser.add_argument("--min_score", default=70, type=int, help="minimum fuzzy matching score threshold")
    parser.add_argument("--bm25_k", default=10, type=int, help="number of BM25 results to retrieve")

    logging.basicConfig(level=logging.INFO)
    log.setLevel(logging.INFO)

    args = parser.parse_args()

    split = args.split

    if args.data_version == "2025":
        import tot_25 as tot
        split += "-2025"
    elif args.data_version == "2024":
        import tot_24 as tot
        split += "-2024"
    else:
        raise ValueError(f"Unknown data version: {args.data_version}")

    tot.register(args.data_path)

    irds_name = "trec-tot:" + split
    dataset = ir_datasets.load(irds_name)

    args = parser.parse_args()
    docs_path = os.path.join(args.docs_path, args.index_name)
    index = os.path.join(args.index_path, args.index_name)
    if args.gather_wikidata_aliases:
        AL_PATH = "/home/wenxin/project/data/id_to_aliases.json"
    else:
        AL_PATH = "./wiki-no-aliases.json"
    if not os.path.exists(index):
        log.info("Creating index!")
        aliases = create_title_index(dataset=dataset,
                                     dest_folder=docs_path,
                                     index=index,
                                     gather_wikidata_aliases=args.gather_wikidata_aliases)
        aliases = {k: list(v) for (k, v) in aliases.items()}
        utils.write_json(aliases, AL_PATH)
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
            print('title:', title, 'matches:', gen_title_to_doc_ids[title])
        else:

            # no exact match, perform retrieval, followed by matching
            res = searcher.search(title, k=BM25_K)

            choices = []
            for doc in res:
                # get the closest alias
                if hasattr(doc, 'docid'):  # for 2024 data
                    docid = doc.docid
                else:  # for 2025 data
                    docid = doc.id
                best_match, score = process.extractOne(title, aliases[docid])
                choices.append(best_match)

            matched = process.extractOne(title, choices)
            if matched is None:
                unmatched.add(title)
                unmatched_props[title] = {
                    "choices": choices
                }
                print('title:', title, 'no match from bm25 search (L290)')
                continue

            matched_title, score = matched
            if score >= MIN_SCORE:
                gen_title_to_doc_ids[title] = resolve(title=title,
                                                      matched_title=matched_title,
                                                      title_to_doc_id=title_to_doc_id,
                                                      aliases=aliases,
                                                      scorer=scorer,
                                                      assert_perfect_score=False)

                if len(gen_title_to_doc_ids[title]) == 1:
                    matches["inexact_1"] += 1
                    print(title, gen_title_to_doc_ids[title])
                else:
                    matches["inexact_n"] += 1
                print('title:', title, 'fuzzy matches:', gen_title_to_doc_ids[title])

            ## try again after removing braces and non alpha numeric characters

            # we need to retain the original titles for mapping it back
            nobr2br = {remove_non_alpha(remove_braces(_)): _ for _ in choices}
            choices_nobr = list(nobr2br.keys())
            matched_nobr = process.extractOne(remove_non_alpha(remove_braces(title)), choices_nobr)
            if matched_nobr is None:
                unmatched.add(title)
                unmatched_props[title] = {
                    "choices": choices,
                    "choices_nobr": choices_nobr,
                    "matched": matched
                }
                print('title:', title, 'no match from matched_nobr (L322)')
                continue

            matched_title_nobr, score_nobr = matched_nobr
            if score_nobr >= MIN_SCORE:
                matched_org_title = nobr2br[matched_title_nobr]

                gen_title_to_doc_ids[title] = resolve(title=title,
                                                      matched_title=matched_org_title,
                                                      title_to_doc_id=title_to_doc_id,
                                                      aliases=aliases,
                                                      scorer=scorer,
                                                      assert_perfect_score=False)

                print('title:', title, 'fuzzy nobr matches:', gen_title_to_doc_ids[title])
            else:
                unmatched.add(title)
                unmatched_props[title] = {
                    "choices": choices,
                    "choices_nobr": choices_nobr,
                    "matched": matched,
                    "matched_nobr": matched_nobr
                }
                print('title:', title, 'no match from matched_nobr (L344)', matched_nobr)
                continue

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
        # qid = query["id"]
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

    if dataset.has_qrels():
        qrel, n_missing = utils.get_qrel(dataset, run)
        metrics = args.metrics.split(",")

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrel, metrics)
        eval_res = evaluator.evaluate(run)

        eval_res_agg = utils.aggregate_pytrec(eval_res, "mean")

        for metric, (mean, std) in eval_res_agg.items():
            log.info(f"{metric:<12}: {mean:.4f} ({std:0.4f})")
    else:
        log.info("dataset has no qrels. no eval performed!")

    # write run file
    run_id = args.run_id
    with open(args.run, "w") as writer:
        for qid, r in run.items():
            for rank, (doc_id, score) in enumerate(sorted(r.items(), key=lambda _: -_[1])):
                writer.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{float(score)}\t{run_id}\n")
