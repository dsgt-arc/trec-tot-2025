#!/bin/bash
#SBATCH -Jedgelist --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G
#SBATCH -t24:00:00 -qinferno -ologs/Jedgelist-%j.out
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -x
cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}
uv run process.py
