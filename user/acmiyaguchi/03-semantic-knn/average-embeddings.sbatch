#!/bin/bash
#SBATCH -Javerage-embeddings --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=192G
#SBATCH -t2:00:00 -qembers -ologs/average-embeddings-%j.out
##SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -xe
export SPARK_LOCAL_DIRS="$TMPDIR"
uv run - <<EOF
# /// script
# dependencies = [
#   "pyspark",
#   "numpy",
# ]
# ///
from pyspark.sql import SparkSession, functions as F
import numpy as np
from pathlib import Path

spark = (
    SparkSession.builder.master("local[24]")
    .config("spark.driver.memory", "190g")
    .getOrCreate()
)
root = Path("~/scratch/trec-tot-2025/data").expanduser()
emb_root = root / "wikipedia-2024-06-bge-m3/data/en"
out_root = root / "enwiki/parquet/bge-m3-avg/v1"

@F.udf(returnType="array<float>")
def avg_vector(vectors: list) -> list:
    return np.mean(np.array(vectors), axis=0).tolist()

emb = spark.read.parquet(emb_root.as_posix())
emb_avg = (
    emb.withColumn("page_id", F.split("id", "_")[0])
    .groupBy("page_id")
    .agg(F.collect_list("embedding").alias("embeddings"))
    .select("page_id", avg_vector("embeddings").alias("embedding"))
)
emb_avg.write.parquet(out_root.as_posix(), mode="overwrite")
EOF
done
