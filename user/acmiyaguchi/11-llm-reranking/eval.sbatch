#!/bin/bash
#SBATCH -Jeval --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t4:00:00 -qembers -ologs/eval-%j.out
#SBATCH --mail-type=START,END,FAIL --mail-user=acmiyaguchi@gatech.edu

source $HOME/trec-tot-2025/.venv/bin/activate
set -ex

script_parent_path=$HOME/trec-tot-2025/user/acmiyaguchi/11-llm-reranking
rerank_path=$HOME/trec-tot-2025/user/wenxinmmb/rerank_module
official_path=$HOME/scratch/trec-tot-2025/data/official
gdrive_path=$HOME/scratch/trec-tot-2025/data/gdrive/data
output_path=$HOME/scratch/trec-tot-2025/results/rerank

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
	$output_path/gemini-2.5-flash-dev3-gemma3-12b-rerank-v2/rerank-results-reformatted.txt \
	$official_path/dev3-2025-qrel.txt
