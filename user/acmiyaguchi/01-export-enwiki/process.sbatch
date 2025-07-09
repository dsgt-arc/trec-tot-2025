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
export BASE_DIR="$HOME/scratch/trec-tot-2025/data/enwiki"
export SIF_FILE="$BASE_DIR/mariadb_latest.sif"
# get slurm launch directory _or_ the directory of this script
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR=$(dirname "$0")
fi
cp $SLURM_SUBMIT_DIR/{z-process.sh,writable-socket.cnf} "$BASE_DIR/"

# Define directories for the process
export CONFIG_FILE="$BASE_DIR/writable-socket.cnf" # Path to new config file
export INIT_DIR="$BASE_DIR/init-scripts" # For scripts and SQL files
export DATA_DIR="$BASE_DIR/mariadb-data"  # For the persistent database
export OUTPUT_DIR="$BASE_DIR/csv-output"  # For the final CSV files

# --- Setup ---
echo "INFO: Cleaning up previous run and setting up directories..."
rm -rf "$DATA_DIR" "$INIT_DIR" # Remove old links and dir
mkdir -p "$INIT_DIR" "$DATA_DIR" "$OUTPUT_DIR" "$BASE_DIR/logs"

echo "INFO: Preparing initialization files by creating symbolic links..."
# Use 'ln -s' to link to the large files instead of copying them
ln -s "$BASE_DIR"/*.sql.gz "$INIT_DIR/"
# The export script is small, so copying is fine
cp "$BASE_DIR/z-process.sh" "$INIT_DIR/"
chmod +x "$INIT_DIR/z-process.sh"

# Use 'apptainer run' to execute the container's default entrypoint
apptainer run \
    --bind "$INIT_DIR:/docker-entrypoint-initdb.d" \
    --bind "$DATA_DIR:/var/lib/mysql" \
    --bind "$OUTPUT_DIR:/output" \
    --bind "$CONFIG_FILE:/etc/mysql/conf.d/z-writable-socket.cnf:ro" \
    --env MARIADB_DATABASE=wikidb \
    --env MARIADB_ROOT_PASSWORD=verysecret \
    "$SIF_FILE"

echo "Job finished. CSV files should be in $OUTPUT_DIR"