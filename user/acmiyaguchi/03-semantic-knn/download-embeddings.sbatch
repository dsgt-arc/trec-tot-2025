#!/bin/bash
#SBATCH -Jdownload-embeddings --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH -t4:00:00 -qembers -ologs/download-%j.out
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

source $HOME/trec-tot-2025/.venv/bin/activate
set -x
BASE_DIR="$HOME/scratch/trec-tot-2025/data"

huggingface-cli download \
    --repo-type dataset \
    --local-dir "$BASE_DIR/wikipedia-2024-06-bge-m3" \
    "Upstash/wikipedia-2024-06-bge-m3"
