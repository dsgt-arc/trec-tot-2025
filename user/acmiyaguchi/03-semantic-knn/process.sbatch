#!/bin/bash
#SBATCH -Jprocess --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=300G
#SBATCH -t6:00:00 -qinferno -ologs/process-%j.out
#SBATCH --mail-type=START,END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -x
cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}
export SPARK_LOCAL_DIRS=$HOME/scratch/spark-tmp
uv run process.py
