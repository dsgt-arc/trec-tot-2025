#!/bin/bash
#SBATCH -Jrerank --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --partition=gpu-l40s --gres=gpu:L40s
#SBATCH -t12:00:00 -qinferno -ologs/rerank-v6-%j.out
#SBATCH --mail-type=START,END,FAIL --mail-user=acmiyaguchi@gatech.edu

source $HOME/trec-tot-2025/.venv/bin/activate
set -ex
# module load ollama/0.6.6
# module load ollama
# uv pip install vllm --torch-backend=auto
hostname
nvidia-smi

# check if we're already serving something over port 8000 and kill if there is
# unfortunately it looks like we keep getting scheduled onto the same machines here
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using port: $PORT"
lsof -i:$PORT && { echo "Port $PORT in use, exiting"; exit 1; }

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
num_queries=${NUM_QUERIES:-100}
batch_rank_end=${BATCH_RANK_END:-100}
MODEL=${MODEL:-"gaunernst/gemma-3-27b-it-qat-compressed-tensors"}
MODEL_SANITIZED=$(echo "$MODEL" | sed 's/[/:]/-/g')
# $ ls $HOME/scratch/trec-tot-2025/data/gdrive/data/shared_retrieval_results
# bge-passage-dense  gemini-2.5-flash  mini-o4  pyterrier-bm25
RETRIEVAL_MODEL=${RETRIEVAL_MODEL:-"pyterrier-bm25"}
DEVSET=${DEVSET:-"dev3"}
OUTPUT_SUFFIX=v6-n${batch_rank_end}-${RETRIEVAL_MODEL}-${DEVSET}-${MODEL_SANITIZED}-rerank
INPUT_PATH=$gdrive_path/shared_retrieval_results/$RETRIEVAL_MODEL/$DEVSET.*

# Simple flock-based semaphore to prevent duplicate jobs
LOCK_DIR="$HOME/scratch/trec-tot-2025/locks"
LOCK_FILE="$LOCK_DIR/${OUTPUT_SUFFIX}.lock"
mkdir -p "$LOCK_DIR"

# Use flock to prevent duplicate experiments
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "Another job is already running this experiment: $OUTPUT_SUFFIX"
    echo "Exiting to avoid wasting GPU resources."
    exit 0
fi

# check if rerank-results already exist and exit early
if [ -f "$output_path/$OUTPUT_SUFFIX/rerank-results-reformatted.txt" ]; then
	echo "Rerank results already exist. Exiting."
	exit 0
fi

# we run the benchmark just to make sure that the model has been downloaded locally
vllm bench throughput \
    --model $MODEL \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy \
    --max-model-len 8092

# function to wait for vllm api
wait_for_vllm() {
    echo "Waiting for vLLM API..."
    for i in {1..50}; do
        if curl -s -w "%{http_code}" -o /dev/null http://localhost:$PORT/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"Hi\",\"max_tokens\":1,\"temperature\":0}" \
            | grep -q "200"; then
            echo "API ready"; return 0
        fi
        [ $i -eq 50 ] && { echo "API timeout"; kill $vllm_pid; exit 1; }
        echo "Attempt $i/50..."; sleep 10
    done
}

# start vllm in the background
# vllm serve --help
vllm serve $MODEL --port $PORT --max-model-len 65536 &
vllm_pid=$!

wait_for_vllm
nvidia-smi

time python $rerank_path/tot_llm_reranking.py \
    --input-trec-run $INPUT_PATH \
    --queries-file $official_path/$DEVSET-2025-queries.jsonl \
    --corpus-file $official_path/trec-tot-2025-corpus.jsonl \
    --offset-file $gdrive_path/corpus-offset-mapping.json \
    --prompt-template-path $script_parent_path/rank_lrl_template.yaml \
    --model $MODEL \
    --save-invocations-history \
    --num-queries $num_queries \
    --batch-rank-end ${BATCH_RANK_END:-100} \
    --output-dir $output_path/$OUTPUT_SUFFIX \
    --api-url http://localhost:$PORT/v1 \
    "$@"

# now kill vllm
kill $vllm_pid

python $rerank_path/rewrite_score.py $output_path/$OUTPUT_SUFFIX/rerank-results.txt

# also write out the current slurm task id to a file
echo $SLURM_JOB_ID > $output_path/$OUTPUT_SUFFIX/slurm_job_id.txt

function run_trec_eval {
	local run_file=$1
	local qrel_file=$2
	trec_eval \
		-m recip_rank \
		-m ndcg_cut.10,100,1000 \
		-m recall.10,100,1000 \
        --complete_rel_info_wanted \
		$qrel_file \
		$run_file
}

# let's create a truncated devset file
truncated=$official_path/$DEVSET-2025-qrel-h${num_queries}.txt
# check if exists
if [ ! -f "$truncated" ]; then
    cat $official_path/$DEVSET-2025-qrel.txt | head -n${num_queries} > $truncated
fi

echo "original results"
run_trec_eval \
	$INPUT_PATH \
	$truncated

echo "reranked results"
run_trec_eval \
	$output_path/$OUTPUT_SUFFIX/rerank-results-reformatted.txt \
	$truncated

exit 0
