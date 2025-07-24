#!/bin/bash
#SBATCH -Jexport-enwiki-stats --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH -t2:00:00 -qembers -ologs/export-enwiki-stats-%j.out
##SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -xe
BASE_DIR="$HOME/scratch/trec-tot-2025/data/enwiki"
OUTPUT_DIR="$BASE_DIR/parquet"
mkdir -p $OUTPUT_DIR

export SPARK_LOCAL_DIRS="$TMPDIR"

for f in $BASE_DIR/parquet/*; do
    TABLE_NAME=$(basename "$f" .csv)
    uv run - <<EOF
# /// script
# dependencies = [
#   "pyspark",
# ]
# ///
from pyspark.sql import SparkSession
spark = (
    SparkSession.builder.master("local[16]")
    .config("spark.driver.memory", "60g")
    .getOrCreate()
)
print("Showing data for table: $TABLE_NAME")
df = spark.read.parquet("$f")
df.printSchema()
df.show(n=5)
print(f"Number of rows: {df.count()}")
EOF
done
