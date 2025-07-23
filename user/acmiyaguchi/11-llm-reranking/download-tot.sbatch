#!/bin/bash
#SBATCH -Jtot-dl --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t2:00:00 -qinferno -ologs/tot-dl-%j.out
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu
set -xeu

path=~/scratch/trec-tot-2025/data/official
mkdir -p $path
aria2c -c -j 3 -x 8 -i download-tot.txt -d $path
