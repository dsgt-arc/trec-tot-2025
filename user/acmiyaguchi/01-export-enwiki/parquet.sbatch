#!/bin/bash
#SBATCH -Jexport-enwiki-parquet --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
#SBATCH -t2:00:00 -qembers -ologs/export-enwiki-parquet-%j.out
##SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -xe
BASE_DIR="$HOME/scratch/trec-tot-2025/data/enwiki"
CSV_DIR="$BASE_DIR/csv"
OUTPUT_DIR="$BASE_DIR/parquet"
mkdir -p $OUTPUT_DIR

if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR=$(dirname "$0")
fi
cd $SLURM_SUBMIT_DIR
export SPARK_LOCAL_DIRS="$TMPDIR"

for f in $CSV_DIR/*.csv; do
    TABLE_NAME=$(basename "$f" .csv)
    uv run parquet.py "$f" "$OUTPUT_DIR/$TABLE_NAME/" --cores 16 --memory 90G
done
