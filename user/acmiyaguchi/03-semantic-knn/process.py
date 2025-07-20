# /// script
# dependencies = [
#   "pyspark",
#   "numpy",
#   "pandas",
#   "pyarrow",
#   "faiss-cpu",
#   "luigi",
#   "typer",
#   "tqdm"
# ]
# ///
import typer
import luigi
from pathlib import Path
from pyspark.sql import SparkSession, functions as F
import numpy as np
import faiss
import pandas as pd
import tqdm

app = typer.Typer()


def get_spark(cores=24, memory="80g"):
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .config("spark.driver.memory", memory)
        .config("spark.driver.maxResultSize", "0")
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


class FaissPCAIndexTask(luigi.Task):
    """We create a FAISS index that uses PCA to reduce the dimensionality of the embeddings.
    This will help with making indexing a bit more efficient.
    """

    input_root = luigi.Parameter()
    output_root = luigi.Parameter()
    dim_input = luigi.IntParameter(default=1024)
    dim_pca = luigi.IntParameter(default=384)
    should_sample = luigi.BoolParameter(default=True)

    def output(self):
        return {
            "pca": luigi.LocalTarget(f"{self.output_root}/pca.bin"),
            "eigenvalues": luigi.LocalTarget(f"{self.output_root}/eigenvalues.npy"),
        }

    def _load(self):
        spark = get_spark()
        df = spark.read.parquet(self.input_root).select("embedding")
        if self.should_sample:
            print("sampling 20% of the data for PCA training")
            df = df.sample(fraction=0.2, seed=42)
        df.explain()
        df = df.toPandas()
        spark.stop()
        return np.stack(df["embedding"].values).astype("float32")

    def run(self):
        X = self._load()

        # normalize for PCA
        faiss.normalize_L2(X)
        pca = faiss.PCAMatrix(self.dim_input, self.dim_pca)
        pca.train(X)

        # write out the PCA matrix
        v = faiss.vector_float_to_array(pca.eigenvalues)
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
        np.save(self.output()["eigenvalues"].path, v)
        faiss.write_VectorTransform(pca, self.output()["pca"].path)


class FaissCosineIndexTask(luigi.Task):
    """We create a FAISS index that uses cosine similarity.
    First we need to normalize the vectors to unit length.
    Then we use the inner product index in order to compute the cosine similarity.
    Because we are projecting using PCA, we will also need to normalize after that projection.
    """

    input_root = luigi.Parameter()
    output_root = luigi.Parameter()
    dim_pca = luigi.IntParameter(default=384)

    def requires(self):
        return {
            "pca": FaissPCAIndexTask(
                input_root=self.input_root,
                output_root=self.output_root,
                dim_pca=self.dim_pca,
            )
        }

    def output(self):
        return {"index": luigi.LocalTarget(f"{self.output_root}/faiss.index")}

    def _load_parts(self):
        parts = sorted(Path(self.input_root).glob("*.parquet"))
        for part in tqdm.tqdm(parts):
            df = pd.read_parquet(part)
            yield (
                np.stack(df["embedding"].values).astype("float32"),
                df["page_id"].values.astype("int64"),
            )

    def _get_index(self):
        pca_task = self.requires()["pca"]
        dim_pca = pca_task.dim_pca
        dim_input = pca_task.dim_input
        pca_matrix = faiss.read_VectorTransform(pca_task.output()["pca"].path)
        # input -> normalize -> pca -> normalize -> index
        index = faiss.IndexIDMap(
            faiss.IndexPreTransform(
                faiss.NormalizationTransform(dim_input, 2.0),
                faiss.IndexPreTransform(
                    pca_matrix,
                    faiss.IndexPreTransform(
                        faiss.NormalizationTransform(dim_pca, 2.0),
                        faiss.IndexFlatIP(dim_pca),
                    ),
                ),
            )
        )
        return index

    def run(self):
        index = self._get_index()
        for X, page_ids in self._load_parts():
            index.add_with_ids(X, page_ids)
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, self.output()["index"].path)


class FaissKNNQueryTask(luigi.Task):
    input_root = luigi.Parameter()
    output_root = luigi.Parameter()
    dim_pca = luigi.IntParameter(default=384)
    k = luigi.IntParameter(default=50)

    def requires(self):
        return {
            "index": FaissCosineIndexTask(
                input_root=self.input_root,
                output_root=self.output_root,
                dim_pca=self.dim_pca,
            )
        }

    def _load_parts(self):
        parts = sorted(Path(self.input_root).glob("*.parquet"))
        for part in tqdm.tqdm(parts):
            df = pd.read_parquet(part)
            yield (
                part.name,
                np.stack(df["embedding"].values).astype("float32"),
                df["page_id"].values.astype("int64"),
            )

    def _get_index(self):
        index_task = self.requires()["index"]
        index = faiss.read_index(index_task.output()["index"].path)
        return index

    def output(self):
        return {"knn": luigi.LocalTarget(f"{self.output_root}/knn/_SUCCESS")}

    def run(self):
        index = self._get_index()

        output_dir = Path(self.output()["knn"].path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        for part_name, X, page_ids in self._load_parts():
            scores, neighbors = index.search(X, self.k)
            results_df = pd.DataFrame(
                {
                    "page_id": page_ids,
                    "neighbors": list(neighbors),
                    "scores": list(scores),
                }
            )
            output_path = output_dir / part_name
            results_df.to_parquet(output_path, index=False)

        with self.output()["knn"].open("w") as f:
            f.write("")


class Workflow(luigi.Task):
    def run(self):
        root = Path("~/scratch/trec-tot-2025/data").expanduser()
        emb_root = root / "wikipedia-2024-06-bge-m3/data/en"
        emb_avg_root = root / "enwiki/parquet/bge-m3-avg/v1"

        yield AverageEmbeddingsTask(
            input_root=str(emb_root),
            output_root=str(emb_avg_root),
        )
        yield FaissKNNQueryTask(
            input_root=str(emb_avg_root),
            output_root=str(root / "enwiki/faiss/bge-m3-avg/v1"),
            dim_pca=384,
            k=50,
        )


@app.command()
def run():
    assert luigi.build([Workflow()], local_scheduler=True)


if __name__ == "__main__":
    app()
