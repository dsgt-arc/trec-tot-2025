# /// script
# dependencies = [
#   "polars",
#   "rustworkx",
#   "tqdm",
#   "luigi",
#   "typer",
#   "contexttimer",
# ]
# ///
from pathlib import Path

import contexttimer
import luigi
import polars as pl
import rustworkx as rx
import tqdm
import typer

app = typer.Typer()


def get_mapping_df(edges: pl.DataFrame) -> pl.DataFrame:
    """Creates a DataFrame mapping page_ids to 0-based indices."""
    src_nodes = edges.select(pl.col("src").alias("id"))
    dst_nodes = edges.select(pl.col("dst").alias("id"))

    # with_row_count is perfect for creating the 0-based contiguous index
    mapping_df = src_nodes.vstack(dst_nodes).unique().sort("id").with_row_index("idx")
    return mapping_df


def load_graph_data(nodes, edges):
    graph = rx.PyDiGraph()
    _ = graph.add_nodes_from(tqdm.tqdm(nodes.sort(by="idx").select("idx").to_series()))
    _ = graph.add_edges_from_no_data(tqdm.tqdm(edges.iter_rows(), total=len(edges)))
    return graph


def load_graph_data_remapped(edges):
    """Load graph data from edges DataFrame with remapped indices."""
    mapping_df = get_mapping_df(edges)
    remapped_edges = (
        edges.join(mapping_df, left_on="src", right_on="id")
        .rename({"idx": "src_idx"})
        .join(mapping_df, left_on="dst", right_on="id")
        .rename({"idx": "dst_idx"})
        .select([pl.col("src_idx").alias("src"), pl.col("dst_idx").alias("dst")])
    )
    nodes = mapping_df.select("idx")
    return load_graph_data(nodes, remapped_edges), mapping_df


class SharedParams:
    edges_path = luigi.Parameter()
    nodes_path = luigi.Parameter()
    output_path = luigi.Parameter()


class ComputePageRank(luigi.Task, SharedParams):
    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        edges = pl.read_parquet(f"{self.edges_path}/*.parquet")
        graph, mapping_df = load_graph_data_remapped(edges)
        with contexttimer.Timer() as t:
            score = rx.pagerank(graph, max_iter=250, tol=1.0e-8)
        print(f"PageRank computed in {t.elapsed:.2f} seconds", flush=True)
        pr_df = (
            pl.DataFrame({"idx": score.keys(), "pagerank": score.values()})
            .join(mapping_df, on="idx", how="inner")
            .select("id", "pagerank")
        )
        pr_df.write_parquet(self.output_path, compression="zstd")


class ComputeHITS(luigi.Task, SharedParams):
    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        edges = pl.read_parquet(f"{self.edges_path}/*.parquet")
        graph, mapping_df = load_graph_data_remapped(edges)
        with contexttimer.Timer() as t:
            hubs, authorities = rx.hits(graph, max_iter=250, tol=1.0e-8)
        print(f"HITS computed in {t.elapsed:.2f} seconds", flush=True)
        hubs_df = pl.DataFrame({"idx": hubs.keys(), "hub": hubs.values()})
        auth_df = pl.DataFrame(
            {"idx": authorities.keys(), "authority": authorities.values()}
        )
        hits_df = (
            hubs_df.join(auth_df, on="idx", how="inner")
            .join(mapping_df, on="idx", how="inner")
            .select("id", "hub", "authority")
        )
        hits_df.write_parquet(self.output_path, compression="zstd")


class ComputeDegreeCentrality(luigi.Task, SharedParams):
    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        edges = pl.read_parquet(f"{self.edges_path}/*.parquet")
        graph, mapping_df = load_graph_data_remapped(edges)
        with contexttimer.Timer() as t:
            in_degrees = rx.in_degree_centrality(graph)
        print(f"Degree computed in {t.elapsed:.2f} seconds", flush=True)
        with contexttimer.Timer() as t:
            out_degrees = rx.out_degree_centrality(graph)
        print(f"Degree computed in {t.elapsed:.2f} seconds", flush=True)
        in_deg_df = pl.DataFrame(
            {
                "idx": in_degrees.keys(),
                "in_degree_centrality": in_degrees.values(),
            }
        )
        out_deg_df = pl.DataFrame(
            {
                "idx": out_degrees.keys(),
                "out_degree_centrality": out_degrees.values(),
            }
        )
        degree_df = (
            in_deg_df.join(out_deg_df, on="idx", how="inner")
            .join(mapping_df, on="idx", how="inner")
            .select("id", "in_degree_centrality", "out_degree_centrality")
        )
        degree_df.write_parquet(self.output_path, compression="zstd")


class ComputeDegree(luigi.Task, SharedParams):
    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        nodes = pl.read_parquet(f"{self.nodes_path}/*.parquet")
        edges = pl.read_parquet(f"{self.edges_path}/*.parquet")
        # just do groupby counts here
        with contexttimer.Timer() as t:
            in_degrees = (
                edges.select(pl.col("dst").alias("id"))
                .group_by("id")
                .agg(pl.count().alias("in_degree"))
            )
        print(f"In-Degree computed in {t.elapsed:.2f} seconds")
        with contexttimer.Timer() as t:
            out_degrees = (
                edges.select(pl.col("src").alias("id"))
                .group_by("id")
                .agg(pl.count().alias("out_degree"))
            )
        print(f"Out-Degree computed in {t.elapsed:.2f} seconds")
        degree_df = (
            nodes.select("id")
            .join(in_degrees, on="id", how="left")
            .join(out_degrees, on="id", how="left")
            .fill_null(0)
        )
        degree_df.write_parquet(self.output_path, compression="zstd")


class Workflow(luigi.Task):
    def run(self):
        trec_root = Path("~/scratch/trec-tot-2025").expanduser()
        dataset_root = trec_root / "data/enwiki/processed"

        suffix = "bge-m3-knn"
        graph_root = dataset_root / "graph/v2" / suffix
        output_root = dataset_root / "centrality/v2" / suffix
        output_root.mkdir(parents=True, exist_ok=True)

        yield [
            ComputePageRank(
                nodes_path=(graph_root / "nodes").as_posix(),
                edges_path=(graph_root / "edges").as_posix(),
                output_path=(output_root / "pagerank.parquet").as_posix(),
            ),
            ComputeHITS(
                nodes_path=(graph_root / "nodes").as_posix(),
                edges_path=(graph_root / "edges").as_posix(),
                output_path=(output_root / "hits.parquet").as_posix(),
            ),
            ComputeDegreeCentrality(
                nodes_path=(graph_root / "nodes").as_posix(),
                edges_path=(graph_root / "edges").as_posix(),
                output_path=(output_root / "degree_centrality.parquet").as_posix(),
            ),
            ComputeDegree(
                nodes_path=(graph_root / "nodes").as_posix(),
                edges_path=(graph_root / "edges").as_posix(),
                output_path=(output_root / "degree.parquet").as_posix(),
            ),
        ]


@app.command()
def run():
    luigi.build([Workflow()], local_scheduler=True)


if __name__ == "__main__":
    app()
