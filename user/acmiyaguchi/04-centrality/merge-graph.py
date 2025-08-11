# /// script
# dependencies = [
#   "pyspark",
#   "numpy",
#   "luigi",
#   "typer",
# ]
# ///
from pyspark.sql import SparkSession, functions as F
import typer
import luigi
from pathlib import Path

app = typer.Typer()


def get_spark(cores=24, memory="190g"):
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .config("spark.driver.memory", memory)
        .config("spark.jars.packages", "graphframes:graphframes:0.8.4-spark3.5-s_2.13")
        .getOrCreate()
    )
    return spark


class MergeGraphs(luigi.Task):
    """Do the same as GenerateEdgeList but use GraphFrames instead."""

    graph_paths = luigi.ListParameter()
    weight_multiplier = luigi.ListParameter()
    default_weight = luigi.FloatParameter(default=1.0)
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

    def load_graph(self, spark, path):
        print(path)
        nodes_path = f"{path}/nodes"
        edges_path = f"{path}/edges"
        v = spark.read.parquet(nodes_path)
        e = spark.read.parquet(edges_path)
        # need this for the knn graph
        if "score" in e.columns and "weight" not in e.columns:
            e = e.withColumnRenamed("score", "weight")
        v.printSchema()
        e.printSchema()
        return GraphFrame(v, e)

    def write_graph(self, g):
        g.edges.explain(extended=True)
        g.edges.write.parquet(
            (Path(self.output_path) / "edges").as_posix(), mode="overwrite"
        )
        g.vertices.explain(extended=True)
        g.vertices.write.parquet(
            (Path(self.output_path) / "nodes").as_posix(), mode="overwrite"
        )

    def run(self):
        spark = get_spark()
        G_set = []

        # load up a bunch of graphs into a list
        for path, mult in zip(self.graph_paths, self.weight_multiplier):
            g = self.load_graph(spark, path)
            # check if there is a default weight
            if "weight" not in g.edges.columns:
                g = GraphFrame(
                    g.vertices,
                    g.edges.withColumn("weight", F.lit(self.default_weight * mult)),
                )
            G_set.append(g)

        # merge the graphs together
        g_prime = G_set[0]
        for g in G_set[1:]:
            g_prime = GraphFrame(
                g_prime.vertices.select("id").union(g.vertices.select("id")),
                g_prime.edges.select("src", "dst", "weight").union(
                    g.edges.select("src", "dst", "weight")
                ),
            )

        # and now aggregate the edges using a sum
        g_prime = GraphFrame(
            g_prime.vertices.select("id").distinct(),
            g_prime.edges.groupBy("src", "dst").agg(F.sum("weight").alias("weight")),
        ).dropIsolatedVertices()
        # drop nodes with no edges, or edges without a node
        self.write_graph(g_prime)


class Workflow(luigi.Task):
    def run(self):
        trec_root = Path("~/scratch/trec-tot-2025").expanduser()
        dataset_root = trec_root / "data/enwiki/processed"
        graph_root = dataset_root / "graph/v2"
        output_path = dataset_root / "graph/v2/merged-v2"

        yield MergeGraphs(
            # NOTE: one of these has a `page_is_redirect` column, but we can ignore this for now
            # and have multiple edges...
            graph_paths=[
                f"{graph_root}/article-one-hop",
                f"{graph_root}/article-meta-two-hop",
                f"{graph_root}/bge-m3-knn-k15",
            ],
            weight_multiplier=[1.0, 0.5, 0.25],
            output_path=output_path.as_posix(),
        )


@app.command()
def run():
    assert luigi.build([Workflow()], local_scheduler=True)


if __name__ == "__main__":
    spark = get_spark()
    from graphframes import GraphFrame

    app()
