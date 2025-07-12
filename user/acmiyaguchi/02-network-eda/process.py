# /// script
# dependencies = [
#   "pyspark",
#   "luigi",
#   "typer",
# ]
# ///
from pyspark.sql import SparkSession, functions as F
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
        return luigi.LocalTarget((Path(self.output_path) / "_SUCCESS").as_posix())

    def run(self):
        spark = get_spark()
        page = spark.read.parquet(self.page_path)
        pagelinks = spark.read.parquet(self.pagelinks_path)
        id2wd = spark.read.parquet(self.mapping_path)

        page_filtered = (
            page.where("page_namespace = 0")
            .where(F.col("page_is_redirect") == "0")
            .join(
                id2wd.select(F.col("page_id").cast("integer"), "wikidata_id"),
                on="page_id",
                how="inner",
            )
        ).cache()

        pagelinks_filtered = (
            pagelinks.where("pl_from_namespace = 0")
            .join(
                page_filtered.select(F.col("page_id").alias("pl_from")),
                on="pl_from",
                how="inner",
            )
            .join(
                page_filtered.select(F.col("page_id").alias("pl_target_id")),
                on="pl_target_id",
                how="inner",
            )
        )
        pagelinks_filtered.repartition(32).write.parquet(
            self.output_path, mode="overwrite"
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
            output_path=(output_root / "edgelist/v1").as_posix(),
        )


@app.command()
def run():
    luigi.build([Workflow()], local_scheduler=True)


if __name__ == "__main__":
    app()
