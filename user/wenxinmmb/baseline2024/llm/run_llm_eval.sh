#!/bin/bash
: <<'END_COMMENT'
for split in dev2 train; do
python oprt_retrieve_score.py   \
    --input_file $DATA_PATH/2025/$split-2025/queries.jsonl   \
    --output_file outputs/scored/$split-gpt-oss-multi.jsonl   \
    --start_line 0   \
    --max_tokens 5000   \
    --temperature 0.0   \
    --model "openai/gpt-oss-120b"   \
    --exclude_response_format

python name_match_score.py \
    --input outputs/scored/$split-gpt-oss-multi.jsonl \
    --split $split-gptoss \
    --index_name llm_title_redirects \
    --run outputs/scored/runs/$split-gptoss-100-10-re.txt \
    --run_id gpt-oss

trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c \
    $DATA_PATH/2025/$split-2025/qrel.txt outputs/scored/runs/$split-gptoss-100-10-re.txt

cp outputs/scored/runs/$split-gptoss-100-10-re.txt /home/wenxin/project/shared_retrieval_results/gpt-oss/$split.txt
cp outputs/scored/runs/$split-gptoss-100-10-re.txt $DATA_PATH/results/$split/gpt-oss.txt

done
END_COMMENT


: <<'END_COMMENT'
python name_match_score.py \
    --input outputs/scored/dev3-o4-mini-single.jsonl \
    --split dev3-100-single \
    --index_name llm_title_redirects \
    --run outputs/scored/runs/dev3-100-o4mini-100-10-re.txt \
    --run_id o4-mini-redirect

python name_match_score.py \
    --input outputs/scored/dev3-gemini-2.5-flash-single.jsonl \
    --split dev3-100-single \
    --index_name llm_title_redirects \
    --run outputs/scored/runs/dev3-100-gemini-100-10-re.txt \
    --run_id gemini-redirect

trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c $DATA_PATH/2025/dev3-2025/qrel-first-100.txt \
    outputs/scored/runs/dev3-100-o4mini-100-10-re.txt

trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c $DATA_PATH/2025/dev3-2025/qrel-first-100.txt \
    outputs/scored/runs/dev3-100-gemini-100-10-re.txt
END_COMMENT

# Credits before: 22.86; after: 21.61
python oprt_retrieve_score.py   \
    --input_file $DATA_PATH/2025/dev3-2025/queries.jsonl   \
    --output_file outputs/scored/dev3-100-o4-mini-multi-v4.jsonl   \
    --start_line 0   \
    --max_lines 100  \
    --max_tokens 5000   \
    --temperature 0.0   \
    --model "openai/o4-mini"

python name_match_score.py \
    --input outputs/scored/dev3-100-o4-mini-multi-v4.jsonl \
    --split dev3-100-o4-mini-v4 \
    --index_name llm_title_redirects \
    --run outputs/scored/runs/dev3-100-o4-mini-100-10-re-v4.txt \
    --run_id o4-mini

trec_eval -m ndcg_cut.10,1000 -m recall.1000 -m recip_rank -c \
    $DATA_PATH/2025/dev3-2025/qrel-first-100.txt outputs/scored/runs/dev3-100-o4-mini-100-10-re-v4.txt