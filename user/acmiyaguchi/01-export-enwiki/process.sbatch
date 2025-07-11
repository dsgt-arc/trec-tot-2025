#!/bin/bash
#SBATCH -Jexport-enwiki --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH -t24:00:00 -qinferno -ologs/export-enwiki-%j.out
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

# --- Setup ---
# Set the directory where your data is located
# Assumes the .sif, .sh, and .sql.gz files are all here

# copy current location of the script to WIKI_DIR

# --- Configuration ---
# Set the base directory where your .sif file and this script are located
export TABLE_NAME=${1?"No table name provided. Usage: sbatch process.sbatch <table_name>"}
export BASE_DIR="$HOME/scratch/trec-tot-2025/data/enwiki"
export SIF_FILE="$BASE_DIR/mariadb_latest.sif"
export TASK_DIR="$BASE_DIR/$TABLE_NAME"

echo "INFO: Cleaning up previous run and setting up directories..."
rm -rf "$TASK_DIR"
mkdir -p "$TASK_DIR"
# get slurm launch directory _or_ the directory of this script
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR=$(dirname "$0")
fi

# Define directories for the process
export CONFIG_FILE="$TASK_DIR/bulk-load.cnf" # Path to new config file
export INIT_DIR="$TASK_DIR/init-scripts" # For scripts and SQL files
export DATA_DIR="$TASK_DIR/mariadb-data"  # For the persistent database
export OUTPUT_DIR="$TASK_DIR/csv-output"  # For the final CSV files

echo "INFO: Preparing initialization files by creating symbolic links..."
cp "$SLURM_SUBMIT_DIR/{10-process.sh}.sh" "$INIT_DIR/"
cp "$SLURM_SUBMIT_DIR/bulk-load.cnf" "$TASK_DIR/bulk-load.cnf"

pigz -dc "$BASE_DIR/$FILENAME" | pv | \
    apptainer run \
        --bind "$INIT_DIR:/docker-entrypoint-initdb.d" \
        --bind "$DATA_DIR:/var/lib/mysql" \
        --bind "$CONFIG_FILE:/etc/mysql/conf.d/z-bulk-load.cnf:ro" \
        --env MARIADB_DATABASE=wikidb \
        --env MARIADB_ROOT_PASSWORD=verysecret \
        "$SIF_FILE"

apptainer run \
    --bind "$INIT_DIR:/docker-entrypoint-initdb.d" \
    --bind "$DATA_DIR:/var/lib/mysql" \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$CONFIG_FILE:/etc/mysql/conf.d/z-bulk-load.cnf:ro" \
    --bind "$BASE_DIR:/base:ro" \
    --env TABLE_NAME="$TABLE_NAME" \
    --env MARIADB_DATABASE=wikidb \
    --env MARIADB_ROOT_PASSWORD=verysecret \
    "$SIF_FILE"

echo "Job finished. CSV files should be in $OUTPUT_DIR"