#!/bin/bash
#SBATCH -Jexport-enwiki --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G
#SBATCH -t24:00:00 -qinferno -ologs/export-enwiki-%j.out
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -x
TABLE_NAME=${1?"No table name provided. Usage: sbatch process.sbatch <table_name>"}
BASE_DIR="$HOME/scratch/trec-tot-2025/data/enwiki"
OUTPUT_DIR="$BASE_DIR/csv"
mkdir -p $OUTPUT_DIR

if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR=$(dirname "$0")
fi
cd $SLURM_SUBMIT_DIR

uv run process.py $BASE_DIR/*$TABLE_NAME.sql.gz "$OUTPUT_DIR/$TABLE_NAME.csv"
