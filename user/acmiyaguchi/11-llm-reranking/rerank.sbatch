#!/bin/bash
#SBATCH -Jrerank --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --gres=gpu:1 --constraint=RTX6000 --cpus-per-task=6 --mem-per-gpu=64G
#SBATCH -t4:00:00 -qembers -ologs/rerank-%j.out
#SBATCH --mail-type=START,END,FAIL --mail-user=acmiyaguchi@gatech.edu

source $HOME/trec-tot-2025/.venv/bin/activate
set -ex
module load ollama

# start ollama in the background
ollama serve &
sleep 10  # wait for ollama to start
ollama_pid=$!
ollama pull gemma3:12b

script_parent_path=$HOME/trec-tot-2025/user/acmiyaguchi/11-llm-reranking
rerank_path=$HOME/trec-tot-2025/user/wenxinmmb/rerank_module
official_path=$HOME/scratch/trec-tot-2025/data/official
gdrive_path=$HOME/scratch/trec-tot-2025/data/gdrive/data
output_path=$HOME/scratch/trec-tot-2025/results/rerank

python $rerank_path/tot_llm_reranking.py \
    --input-trec-run $gdrive_path/shared_retrieval_results/gemini-2.5-flash/dev3.run \
    --queries-file $official_path/dev3-2025-queries.jsonl \
    --corpus-file $official_path/trec-tot-2025-corpus.jsonl \
    --offset-file $gdrive_path/corpus-offset-mapping.json \
    --prompt-template-path $script_parent_path/rank_lrl_template.yaml \
    --output-dir $output_path/gemini-2.5-flash-dev3-gemma3-12b-rerank

python $rerank_path/rewrite_score.py \
    $output_path/gemini-2.5-flash-dev3-gemma3-12b-rerank/rerank-results.txt

# now kill ollama and exit
kill $ollama_pid
exit 0
