#!/bin/bash
#SBATCH -Jstats --account=paceship-dsgt_trec2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH -t2:00:00 -qembers -ologs/stats-%j.out
##SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -xe
export SPARK_LOCAL_DIRS="$TMPDIR"

SUFFIX=${1:-"v2.2/bge-m3-knn-k15"}

uv run - <<EOF
# /// script
# dependencies = [
#   "pyspark", "numpy"
# ]
# ///
from pyspark.sql import SparkSession
from pathlib import Path
import pyspark.sql.functions as F
import numpy as np

spark = (
    SparkSession.builder.master("local[16]")
    .config("spark.driver.memory", "60g")
    .getOrCreate()
)
trec_root = Path("~/scratch/trec-tot-2025").expanduser()
dataset_root = trec_root / "data/enwiki/processed"
parquet_root = trec_root / "data/enwiki/parquet"

suffix = "${SUFFIX}"
page = spark.read.parquet((parquet_root / "page").as_posix())
nodes = spark.read.parquet((dataset_root / "graph/v2/bge-m3-knn/nodes").as_posix())
pagerank = spark.read.parquet(
    (dataset_root / f"centrality/{suffix}/pagerank.parquet").as_posix()
)
degree_centrality = spark.read.parquet(
    (dataset_root / f"centrality/{suffix}/degree_centrality.parquet").as_posix()
)
degree = spark.read.parquet(
    (dataset_root / f"centrality/{suffix}/degree.parquet").as_posix()
)
df = (
    nodes.join(
        page.select(F.col("page_id").alias("id"), F.col("page_title").alias("title")),
        on="id",
        how="inner",
    )
    .join(pagerank, on="id", how="inner")
    .join(degree_centrality, on="id", how="inner")
    .join(degree, on="id", how="inner")
).cache()
df.printSchema()
df.show(truncate=False)
df.describe().show(truncate=False)

for col in df.columns:
    if col in ["id", "title"]:
        continue
    df.select("id", "title", col).orderBy(col, ascending=False).show(truncate=False)
    df.select("id", "title", col).orderBy(col, ascending=True).show(truncate=False)
    # for each metric, we should also use approximate quantiles in 10% increments to see the cumulative distribution
    print("Approximate quantiles for", col)
    p = list(np.linspace(0, 1, 21))
    q = df.select(col).approxQuantile(col, p, 0.05)
    for i, quantile in enumerate(q):
        print(f"Quantile {p[i]:.2f}: {quantile}")

EOF
