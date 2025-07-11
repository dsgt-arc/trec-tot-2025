#!/bin/bash
set -e
export BASE_DIR="$HOME/scratch/trec-tot-2025/data/enwiki"
mkdir -p $BASE_DIR
cd $BASE_DIR
apptainer pull docker://mariadb:latest

