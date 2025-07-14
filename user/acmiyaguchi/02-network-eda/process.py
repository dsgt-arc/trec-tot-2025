# /// script
# dependencies = [
#   "pyspark",
#   "luigi",
#   "typer",
# ]
# ///
from pyspark.sql import SparkSession, functions as F, Window
import typer
import luigi
import json
from pathlib import Path

app = typer.Typer()


def get_spark(cores=16, memory="124g"):
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .config("spark.driver.memory", memory)
        .getOrCreate()
    )
    return spark


class ID2WikiData6m(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget((Path(self.output_path) / "_SUCCESS").as_posix())

    def run(self):
        data = json.loads(Path(self.input_path).read_text())
        id2wd = (
            get_spark()
            .createDataFrame(data.items(), schema=["page_id", "wikidata_id"])
            .repartition(32)
            .select(F.col("page_id").cast("integer"), "wikidata_id")
        )
        id2wd.printSchema()
        id2wd.repartition(32).write.parquet(self.output_path, mode="overwrite")


class GenerateEdgeList(luigi.Task):
    mapping_path = luigi.Parameter()
    page_path = luigi.Parameter()
    pagelinks_path = luigi.Parameter()
    output_path = luigi.Parameter()

    def output(self):
        return {
            "edges": luigi.LocalTarget(
                (Path(self.output_path) / "edges/_SUCCESS").as_posix()
            ),
            "nodes": luigi.LocalTarget(
                (Path(self.output_path) / "nodes/_SUCCESS").as_posix()
            ),
        }

    def run(self):
        spark = get_spark()
        page = spark.read.parquet(self.page_path)
        pagelinks = spark.read.parquet(self.pagelinks_path)
        id2wd = spark.read.parquet(self.mapping_path)

        nodes = (
            page.where("page_namespace = 0")
            .where(F.col("page_is_redirect") == "0")
            .join(
                id2wd.select(F.col("page_id").cast("integer"), "wikidata_id"),
                on="page_id",
                how="inner",
            )
            .select("page_id", "wikidata_id")
            .distinct()
            .withColumn(
                "node_id",
                F.row_number().over(Window.orderBy("page_id")) - 1,
            )
            .orderBy("node_id")
        )
        nodes.repartition(8).write.parquet(
            (Path(self.output_path) / "nodes").as_posix(), mode="overwrite"
        )
        nodes = spark.read.parquet(
            (Path(self.output_path) / "nodes").as_posix()
        ).cache()

        edges = (
            pagelinks.where("pl_from_namespace = 0")
            .join(
                nodes.select(
                    F.col("page_id").alias("pl_from"),
                    F.col("node_id").alias("src"),
                ),
                on="pl_from",
                how="inner",
            )
            .join(
                nodes.select(
                    F.col("page_id").alias("pl_target_id"),
                    F.col("node_id").alias("dst"),
                ),
                on="pl_target_id",
                how="inner",
            )
            .select("src", "dst")
            .distinct()
            .orderBy("src", "dst")
        )
        edges.repartition(32).write.parquet(
            (Path(self.output_path) / "edges").as_posix(), mode="overwrite"
        )


class Workflow(luigi.Task):
    def run(self):
        trec_root = Path("~/scratch/trec-tot-2025").expanduser()
        parquet_root = trec_root / "data/enwiki/parquet"
        gdrive_root = trec_root / "data/gdrive/data"
        output_root = trec_root / "data/enwiki/processed"

        id2wd_task = ID2WikiData6m(
            input_path=(gdrive_root / "id2wikidataid_6m.json").as_posix(),
            output_path=(output_root / "id2wikidataid_6m/v1").as_posix(),
        )
        yield id2wd_task

        yield GenerateEdgeList(
            mapping_path=(output_root / "id2wikidataid_6m/v1").as_posix(),
            page_path=(parquet_root / "page").as_posix(),
            pagelinks_path=(parquet_root / "pagelinks").as_posix(),
            output_path=(output_root / "graph/v1").as_posix(),
        )


@app.command()
def run():
    luigi.build([Workflow()], local_scheduler=True)


if __name__ == "__main__":
    app()
