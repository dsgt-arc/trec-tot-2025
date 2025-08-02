Commands

```
DATA_PATH=/home/wenxin/project/data/2025

$ python open_router_basic.py --input_file $DATA_PATH/dev3-2025/queries.jsonl --output_file output/gmn-flash-0801/dev3-org.jsonl --max_tokens 5000 --temperature 0 --start-line 0 --max-lines 100

$ python llm_match_name.py --split dev3 --data_path $DATA_PATH --index_name llm_title_alias --gather_wikidata_aliases --input output/gmn-flash-0801/dev3-org.jsonl --run output/gmn-flash-0801/dev3-org.txt --run_id gmn_alias 

$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c /home/wenxin/project/data/2025/dev3-2025/qrel-first-100.txt output/gmn-flash-0801/dev3-org.txt
```

For generated query set
```
$ python open_router_basic.py --input_file $DATA_PATH/dev3-g1-2025/queries.jsonl --output_file output/gmn-flash-0801/dev3-g1.jsonl --max_tokens 5000 --temperature 0 --start-line 0 --max-lines 100

$ python llm_match_name.py --split dev3 --data_path $DATA_PATH --index_name llm_title_alias --gather_wikidata_aliases --input output/gmn-flash-0801/dev3-g1.jsonl --run output/gmn-flash-0801/dev3-g1.txt --run_id gmn_alias 

$ trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c /home/wenxin/project/data/2025/dev3-2025/qrel-first-100.txt output/gmn-flash-0801/dev3-g1.txt
```
