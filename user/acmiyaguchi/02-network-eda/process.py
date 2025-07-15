# /// script
# dependencies = [
#   "pyspark",
#   "numpy",
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


def get_spark(cores=24, memory="190g", shuffle_partitions=2000):
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .config("spark.driver.memory", memory)
        .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.13")
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
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


class ArticleOneHop(luigi.Task):
    """Do the same as GenerateEdgeList but use GraphFrames instead."""

    page_path = luigi.Parameter()
    links_path = luigi.Parameter()
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

    def load_graph(self):
        spark = get_spark()
        page = spark.read.parquet(self.page_path)
        pagelinks = spark.read.parquet(self.links_path)

        v = page.select(
            F.col("page_id").alias("id"),
            "page_namespace",
            (F.col("page_is_redirect") != "0").alias("page_is_redirect"),
        )
        e = pagelinks.select(
            F.col("pl_from").alias("src"), F.col("pl_target_id").alias("dst")
        ).distinct()

        return GraphFrame(v, e)

    def write_graph(self, g):
        g.edges.explain(extended=True)
        g.edges.repartition(32).write.parquet(
            (Path(self.output_path) / "edges").as_posix(), mode="overwrite"
        )
        g.vertices.explain(extended=True)
        g.vertices.repartition(8).write.parquet(
            (Path(self.output_path) / "nodes").as_posix(), mode="overwrite"
        )

    def run(self):
        g = self.load_graph().cache()
        motif_edges = self.find_motif(g)

        g_prime = GraphFrame(g.vertices, motif_edges).dropIsolatedVertices().cache()
        self.write_graph(g_prime)
        g.unpersist()
        g_prime.unpersist()

    def find_motif(self, g):
        motif = (
            g.find("(a)-[e]->(b)")
            .where("a.id != b.id")
            .where("a.page_namespace = 0 and a.page_is_redirect = false")
            .where("b.page_namespace = 0 and b.page_is_redirect = false")
        )
        return motif.select("e.*", F.lit(1).alias("weight")).distinct()


class ArticleMetaTwoHop(ArticleOneHop):
    def find_motif(self, g):
        motif = (
            g.find("(a)-[]->(b); (b)-[]->(c)")
            .where("a.id != c.id")
            .where("a.page_namespace = 0 and a.page_is_redirect = false")
            .where("b.page_namespace != 0 or b.page_is_redirect = true")
            .where("c.page_namespace = 0 and c.page_is_redirect = false")
        )
        return (
            motif.select(F.col("a.id").alias("src"), F.col("c.id").alias("dst"))
            .groupBy("src", "dst")
            .agg(F.count("*").alias("weight"))
        )


class ArticleSharedTarget(ArticleOneHop):
    def find_motif(self, g):
        motif = (
            g.find("(a)-[]->(b); (c)-[]->(b)")
            .where("a.id != c.id")
            .where("a.page_namespace = 0 and a.page_is_redirect = false")
            .where("c.page_namespace = 0 and c.page_is_redirect = false")
        )
        # we keep the page_namespace, in case we want to only look at
        # two hops through articles, or two hops through meta.
        # in this case we consider all two hops.
        return (
            motif.select(
                F.col("a.id").alias("src"),
                F.col("c.id").alias("dst"),
                F.col("b.page_namespace").alias("page_namespace"),
            )
            .groupBy("src", "dst", "page_namespace")
            .agg(F.count("*").alias("weight"))
        )


class ArticleCategorySharedTarget(ArticleSharedTarget):
    def load_graph(self):
        spark = get_spark()
        page = spark.read.parquet(self.page_path)
        categorylinks = spark.read.parquet(self.links_path)

        v = page.select(
            F.col("page_id").alias("id"),
            "page_namespace",
            (F.col("page_is_redirect") != "0").alias("page_is_redirect"),
        )
        e = categorylinks.select(
            F.col("cl_from").alias("src"), F.col("cl_target_id").alias("dst")
        ).distinct()

        return GraphFrame(v, e)


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

        yield [
            GenerateEdgeList(
                mapping_path=(output_root / "id2wikidataid_6m/v1").as_posix(),
                page_path=(parquet_root / "page").as_posix(),
                pagelinks_path=(parquet_root / "pagelinks").as_posix(),
                output_path=(output_root / "graph/v1").as_posix(),
            ),
            ArticleOneHop(
                page_path=(parquet_root / "page").as_posix(),
                links_path=(parquet_root / "pagelinks").as_posix(),
                output_path=(output_root / "graph/v2/article-one-hop").as_posix(),
            ),
            ArticleMetaTwoHop(
                page_path=(parquet_root / "page").as_posix(),
                links_path=(parquet_root / "pagelinks").as_posix(),
                output_path=(output_root / "graph/v2/article-meta-two-hop").as_posix(),
            ),
            ArticleSharedTarget(
                page_path=(parquet_root / "page").as_posix(),
                links_path=(parquet_root / "pagelinks").as_posix(),
                output_path=(output_root / "graph/v2/article-shared-target").as_posix(),
            ),
            ArticleCategorySharedTarget(
                page_path=(parquet_root / "page").as_posix(),
                links_path=(parquet_root / "categorylinks").as_posix(),
                output_path=(
                    output_root / "graph/v2/article-category-shared-target"
                ).as_posix(),
            ),
        ]


@app.command()
def run():
    luigi.build([Workflow()], local_scheduler=True)


if __name__ == "__main__":
    spark = get_spark()
    from graphframes import GraphFrame

    app()
