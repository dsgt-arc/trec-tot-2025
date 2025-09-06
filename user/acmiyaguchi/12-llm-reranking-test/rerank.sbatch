#!/bin/bash
#SBATCH -Jrerank --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --partition=gpu-l40s --gres=gpu:L40s
#SBATCH -t12:00:00 -qinferno -ologs/rerank-v7-%j.out
#SBATCH --mail-type=START,END,FAIL --mail-user=acmiyaguchi@gatech.edu
##SBATCH --exclude=atl1-1-01-010-35-0

source $HOME/trec-tot-2025/.venv/bin/activate
set -ex
hostname
nvidia-smi

PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using port: $PORT"
lsof -i:$PORT && { echo "Port $PORT in use, exiting"; exit 1; }

script_parent_path=$HOME/trec-tot-2025/user/acmiyaguchi/11-llm-reranking
rerank_path=$HOME/trec-tot-2025/user/wenxinmmb/rerank_module
official_path=$HOME/scratch/trec-tot-2025/data/official
gdrive_path=$HOME/scratch/trec-tot-2025/data/gdrive/data
output_path=$HOME/scratch/trec-tot-2025/results/rerank

start_query_index=${START_QUERY_INDEX?}
num_queries=${NUM_QUERIES:-50}
batch_rank_end=${BATCH_RANK_END:-500}
DEVSET=${DEVSET:-test-2025}
MODEL=${MODEL:-"gaunernst/gemma-3-27b-it-qat-compressed-tensors"}
# ls $HOME/scratch/trec-tot-2025/data/gdrive/data/shared_retrieval_results/2025-test
# comb_200_gemini_bm25_bge.txt  Description.docx  lambdamart-v5-all-1000.txt
RETRIEVAL_MODEL=${RETRIEVAL_MODEL?}
OUTPUT_SUFFIX=v7/$RETRIEVAL_MODEL/${DEVSET}-${batch_rank_end}-${num_queries}-${start_query_index}
INPUT_PATH=$gdrive_path/shared_retrieval_results/2025-test/$RETRIEVAL_MODEL.*

# Simple flock-based semaphore to prevent duplicate jobs
LOCK_DIR="$HOME/scratch/trec-tot-2025/locks"
LOCK_FILE="$LOCK_DIR/${OUTPUT_SUFFIX}.lock"
mkdir -p "$(dirname "$LOCK_FILE")"

# Use flock to prevent duplicate experiments
# Let bash assign an available file descriptor automatically
exec {lock_fd}>"$LOCK_FILE"
if ! flock -n $lock_fd; then
    echo "Another job is already running this experiment: $OUTPUT_SUFFIX"
    echo "Exiting to avoid wasting GPU resources."
    exit 0
fi
# The lock stays active until the script exits
# now trap exit to remove the lock file
trap 'rm -f "$LOCK_FILE"' EXIT

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
    --queries-file $official_path/$DEVSET-queries.jsonl \
    --corpus-file $official_path/trec-tot-2025-corpus.jsonl \
    --offset-file $gdrive_path/corpus-offset-mapping.json \
    --prompt-template-path $script_parent_path/rank_lrl_template.yaml \
    --model $MODEL \
    --save-invocations-history \
    --start-query-index $(( start_query_index )) \
    --num-queries $num_queries \
    --batch-rank-end ${batch_rank_end} \
    --output-dir $output_path/$OUTPUT_SUFFIX \
    --api-url http://localhost:$PORT/v1 \
    "$@"

# now kill vllm
kill $vllm_pid

python $rerank_path/rewrite_score.py $output_path/$OUTPUT_SUFFIX/rerank-results.txt

# also write out the current slurm task id to a file
echo $SLURM_JOB_ID > $output_path/$OUTPUT_SUFFIX/slurm_job_id.txt
