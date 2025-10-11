#!/bin/bash
#SBATCH -Jsbatch --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=320G
#SBATCH -t12:00:00 -qinferno -ologs/ppr-%j.out
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -x
cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}
uv run ppr.py --workers 3
