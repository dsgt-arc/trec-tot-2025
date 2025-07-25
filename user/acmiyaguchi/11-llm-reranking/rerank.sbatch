#!/bin/bash
#SBATCH -Jrerank --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --gres=gpu:1 --constraint=RTX6000 --cpus-per-task=6 --mem-per-gpu=64G
#SBATCH -t4:00:00 -qembers -ologs/rerank-%j.out
#SBATCH --mail-type=START,END,FAIL --mail-user=acmiyaguchi@gatech.edu

source $HOME/trec-tot-2025/.venv/bin/activate
set -ex
# module load ollama/0.6.6
module load ollama

MODEL=${MODEL:-"gemma3:12b"}

# start ollama in the background
ollama serve &
sleep 10  # wait for ollama to start
ollama_pid=$!
ollama pull $MODEL

script_parent_path=$HOME/trec-tot-2025/user/acmiyaguchi/11-llm-reranking
rerank_path=$HOME/trec-tot-2025/user/wenxinmmb/rerank_module
official_path=$HOME/scratch/trec-tot-2025/data/official
gdrive_path=$HOME/scratch/trec-tot-2025/data/gdrive/data
output_path=$HOME/scratch/trec-tot-2025/results/rerank

OUTPUT_SUFFIX=gemini-2.5-flash-dev3-gemma3-12b-rerank-v3.1
# check if rerank-results already exist and exit early
if [ -f "$output_path/$OUTPUT_SUFFIX/rerank-results.txt" ]; then
	echo "Rerank results already exist. Exiting."
	exit 0
fi

python $rerank_path/tot_llm_reranking.py \
    --input-trec-run $gdrive_path/shared_retrieval_results/gemini-2.5-flash/dev3.run \
    --queries-file $official_path/dev3-2025-queries.jsonl \
    --corpus-file $official_path/trec-tot-2025-corpus.jsonl \
    --offset-file $gdrive_path/corpus-offset-mapping.json \
    --prompt-template-path $script_parent_path/rank_lrl_template.yaml \
    --model $MODEL \
    --save-invocations-history \
    --num-queries 100 \
    --output-dir $output_path/$OUTPUT_SUFFIX

# now kill ollama
kill $ollama_pid

python $rerank_path/rewrite_score.py $output_path/$OUTPUT_SUFFIX/rerank-results.txt

function run_trec_eval {
	local run_file=$1
	local qrel_file=$2
	trec_eval \
		-m ndcg_cut.10,1000 \
		-m recall.1000 \
		-m recip_rank \
		$qrel_file \
		$run_file
}

echo "original results"
run_trec_eval \
	$gdrive_path/shared_retrieval_results/gemini-2.5-flash/dev3.run \
	$official_path/dev3-2025-qrel.txt

echo "reranked results"
run_trec_eval \
	$output_path/$OUTPUT_SUFFIX/rerank-results-reformatted.txt \
	$official_path/dev3-2025-qrel.txt

exit 0
