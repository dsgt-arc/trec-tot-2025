#!/bin/bash
#SBATCH -Jmerge-graph --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=192G
#SBATCH -t4:00:00 -qembers -ologs/merge-graph-%j.out
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -x
cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}
uv run merge-graph.py
