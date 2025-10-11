#!/bin/bash
#SBATCH -Jbench --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --gpus=1 -pgpu-l40s --cpus-per-task=8
#SBATCH -t4:00:00 -ologs/bench-%j.out
#SBATCH --mail-type=START,END,FAIL --mail-user=acmiyaguchi@gatech.edu

source $HOME/trec-tot-2025/.venv/bin/activate
set -ex

# uv pip install vllm --torch-backend=auto

# gemma3:12b
# mistral-nemo:12b
# qwen3:14b
# google/gemma-3-12b-it
# gaunernst/gemma-3-12b-it-qat-compressed-tensors
MODEL=${MODEL:-"gaunernst/gemma-3-27b-it-qat-compressed-tensors"}

nvidia-smi
vllm bench throughput \
    --model $MODEL \
    --tensor-parallel-size 1 \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy \
    --max-model-len 8092
