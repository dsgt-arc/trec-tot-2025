#!/bin/bash
#SBATCH -Jrerank --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --gpus=1 -pgpu-l40s --cpus-per-task=8
#SBATCH -t8:00:00 -qinferno -ologs/rerank-%j.out
#SBATCH --mail-type=START,END,FAIL --mail-user=acmiyaguchi@gatech.edu

source $HOME/trec-tot-2025/.venv/bin/activate
set -ex
# module load ollama/0.6.6
# module load ollama
# uv pip install vllm --torch-backend=auto

script_parent_path=$HOME/trec-tot-2025/user/acmiyaguchi/11-llm-reranking
rerank_path=$HOME/trec-tot-2025/user/wenxinmmb/rerank_module
official_path=$HOME/scratch/trec-tot-2025/data/official
gdrive_path=$HOME/scratch/trec-tot-2025/data/gdrive/data
output_path=$HOME/scratch/trec-tot-2025/results/rerank

# gemma3:12b
# mistral-nemo:12b
# qwen3:14b
# google/gemma-3-12b-it
# gaunernst/gemma-3-12b-it-qat-compressed-tensors
MODEL=${MODEL:-"google/gemma-3-12b-it"}
MODEL_SANITIZED=$(echo "$MODEL" | sed 's/[/:]/-/g')
# $ ls $HOME/scratch/trec-tot-2025/data/gdrive/data/shared_retrieval_results
# bge-passage-dense  gemini-2.5-flash  mini-o4  pyterrier-bm25
RETRIEVAL_MODEL=${RETRIEVAL_MODEL:-"pyterrier-bm25"}
DEVSET=${DEVSET:-"dev3"}
OUTPUT_SUFFIX=v2-${RETRIEVAL_MODEL}-${DEVSET}-${MODEL_SANITIZED}-rerank
INPUT_PATH=$gdrive_path/shared_retrieval_results/$RETRIEVAL_MODEL/$DEVSET.run

# check if rerank-results already exist and exit early
if [ -f "$output_path/$OUTPUT_SUFFIX/rerank-results.txt" ]; then
	echo "Rerank results already exist. Exiting."
	exit 0
fi

vllm bench throughput \
    --model $MODEL \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy \
    --max-model-len 8092

# start vllm in the background
# vllm serve --help
vllm serve $MODEL &
sleep 10  # wait for vllm to start
vllm_pid=$!


python $rerank_path/tot_llm_reranking.py \
    --input-trec-run $INPUT_PATH \
    --queries-file $official_path/$DEVSET-2025-queries.jsonl \
    --corpus-file $official_path/trec-tot-2025-corpus.jsonl \
    --offset-file $gdrive_path/corpus-offset-mapping.json \
    --prompt-template-path $script_parent_path/rank_lrl_template.yaml \
    --model $MODEL \
    --save-invocations-history \
    --num-queries 100 \
    --output-dir $output_path/$OUTPUT_SUFFIX \
    --api-url http://localhost:8000/v1

# now kill vllm
kill $vllm_pid

python $rerank_path/rewrite_score.py $output_path/$OUTPUT_SUFFIX/rerank-results.txt

function run_trec_eval {
	local run_file=$1
	local qrel_file=$2
	trec_eval \
		-m ndcg_cut.10,1000 \
		-m recall.1000 \
		-m recip_rank \
        -c \
		$qrel_file \
		$run_file
}

echo "original results"
run_trec_eval \
	$INPUT_PATH \
	$official_path/$DEVSET-2025-qrel.txt

echo "reranked results"
run_trec_eval \
	$output_path/$OUTPUT_SUFFIX/rerank-results-reformatted.txt \
	$official_path/$DEVSET-2025-qrel.txt

exit 0
