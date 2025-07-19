# /// script
# dependencies = [
#   "polars",
#   "numpy",
#   "faiss-cpu",
#   "luigi",
#   "typer",
# ]
# ///
import typer
import luigi
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
import numpy as np

app = typer.Typer()


def get_spark(cores=24, memory="190g"):
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .config("spark.driver.memory", memory)
        .getOrCreate()
    )
    return spark


class AverageEmbeddingsTask(luigi.Task):
    input_root = luigi.Parameter()
    output_root = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f"{self.output_root}/_SUCCESS")

    def run(self):
        spark = get_spark()

        @F.udf(returnType="array<float>")
        def avg_vector(vectors: list) -> list:
            return np.mean(np.array(vectors), axis=0).tolist()

        emb = spark.read.parquet(self.input_root)
        emb_avg = (
            emb.withColumn("page_id", F.split("id", "_")[0])
            .groupBy("page_id")
            .agg(F.collect_list("embedding").alias("embeddings"))
            .select("page_id", avg_vector("embeddings").alias("embedding"))
        )
        emb_avg.write.parquet(self.output_root, mode="overwrite")
        spark.stop()


class FaissCosineIndexTask(luigi.Task):
    """We create a FAISS index that uses cosine similarity.
    First we need to normalize the vectors to unit length.
    Then we use the inner product index in order to compute the cosine similarity.
    Because we are projecting using PCA, we will also need to normalize after that projection.
    """

    def output(self):
        return luigi.LocalTarget("faiss_cosine_index_output.json")

    def run(self):
        raise NotImplementedError()


class Workflow(luigi.Task):
    def run(self):
        root = Path("~/scratch/trec-tot-2025/data").expanduser()
        emb_root = root / "wikipedia-2024-06-bge-m3/data/en"
        emb_avg_root = root / "enwiki/parquet/bge-m3-avg/v1"

        yield AverageEmbeddingsTask(
            input_root=str(emb_root),
            output_root=str(emb_avg_root),
        )


@app.command()
def run():
    assert luigi.build([Workflow()], local_scheduler=True)


if __name__ == "__main__":
    app()
