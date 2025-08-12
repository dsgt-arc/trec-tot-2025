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


def load_graph_data(nodes: pl.DataFrame, edges: pl.DataFrame) -> rx.PyDiGraph:
    graph = rx.PyDiGraph()
    _ = graph.add_nodes_from(
        tqdm.tqdm(nodes.sort(by="idx").select("idx").to_series(), mininterval=10)
    )
    _ = graph.add_edges_from(
        tqdm.tqdm(edges.iter_rows(), total=len(edges), mininterval=10)
    )
    return graph


def load_graph_data_remapped(edges: pl.DataFrame) -> tuple[rx.PyDiGraph, pl.DataFrame]:
    """Load graph data from edges DataFrame with remapped indices."""
    mapping_df = get_mapping_df(edges)
    remapped_edges = (
        edges.join(mapping_df, left_on="src", right_on="id")
        .rename({"idx": "src_idx"})
        .join(mapping_df, left_on="dst", right_on="id")
        .rename({"idx": "dst_idx"})
        .select(
            [
                pl.col("src_idx").alias("src"),
                pl.col("dst_idx").alias("dst"),
                pl.col("score").alias("weight"),
            ]
        )
    )
    nodes = mapping_df.select("idx")
    return load_graph_data(nodes, remapped_edges), mapping_df


class SharedParams:
    run_path = luigi.Parameter()
    edges_path = luigi.Parameter()
    output_path = luigi.Parameter()
    shard_index = luigi.IntParameter(description="shard index for parallel processing")
    total_shards = luigi.IntParameter(description="Total number of shards to process")


class RerankPersonalizedPageRank(luigi.Task, SharedParams):
    def output(self):
        return [
            luigi.LocalTarget(f"{self.output_path}/{qid}.parquet")
            for qid in self._sharded_qid(
                self._load_run(), self.shard_index, self.total_shards
            )
        ]

    def _load_run(self) -> pl.DataFrame:
        cols = ["qid", "Q0", "docid", "rank", "score", "run_name"]
        rundf = pl.read_csv(
            self.run_path,
            separator="\t",
            has_header=False,
            new_columns=cols,
        )
        return rundf

    def _sharded_qid(
        self, rundf: pl.DataFrame, shard_index: int, total_shards: int
    ) -> list[int]:
        """Selects a subset of queries based on the shard index.

        Uses module of the sorted qids to determine which queries to include in this shard.
        """
        qids = sorted(rundf.select("qid").unique().to_series())
        out_qids = []
        for i, qid in enumerate(qids):
            if i % total_shards == shard_index:
                out_qids.append(qid)
        return out_qids

    def _compute_ppr(self, graph, mapping_df, rundf, qid):
        output_path = Path(self.output_path) / f"{qid}.parquet"
        if output_path.exists():
            return
        subset = rundf.filter(pl.col("qid") == qid)
        mapped_subset = subset.select(pl.col("docid").alias("id")).join(
            mapping_df, on="id", how="inner"
        )
        dim = mapped_subset.shape[0]
        personalization = {
            idx: 1.0 / dim for idx in mapped_subset.select("idx").to_series()
        }
        ppr = rx.pagerank(
            graph, personalization=personalization, tol=1.0e-8, weight_fn=lambda e: e
        )
        score_df = pl.DataFrame({"idx": ppr.keys(), "score": ppr.values()}).join(
            mapped_subset, on="idx", how="inner"
        )
        subset_reranked = (
            subset.drop("score")
            .join(
                score_df.select(pl.col("id").alias("docid"), pl.col("score")),
                on="docid",
                how="inner",
            )
            .select(subset.columns)
        ).sort("score", descending=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subset_reranked.write_parquet(output_path, compression="zstd")

    def run(self):
        rundf = self._load_run()
        edges = pl.read_parquet(f"{self.edges_path}/*.parquet")
        graph, mapping_df = load_graph_data_remapped(edges)
        for qid in tqdm.tqdm(
            self._sharded_qid(rundf, self.shard_index, self.total_shards)
        ):
            self._compute_ppr(graph, mapping_df, rundf, qid)


class Workflow(luigi.Task):
    def run(self):
        trec_root = Path("~/scratch/trec-tot-2025").expanduser()
        dataset_root = trec_root / "data/enwiki/processed"
        gdrive_path = trec_root / "data/gdrive/data"
        total_shards = 48

        tasks = []

        for suffix in ["bge-m3-knn-k15", "merged-v2"]:
            graph_root = dataset_root / "graph/v2" / suffix
            output_root = dataset_root / "reranked/v2.1" / suffix
            output_root.mkdir(parents=True, exist_ok=True)

            tasks.extend(
                [
                    RerankPersonalizedPageRank(
                        run_path=f"{gdrive_path}/shared_retrieval_results/gemini-2.5-flash/dev3.run",
                        edges_path=f"{graph_root}/edges",
                        output_path=f"{output_root}/ppr",
                        shard_index=shard_index,
                        total_shards=total_shards,
                    )
                    for shard_index in range(total_shards)
                ]
            )
        yield tasks


@app.command()
def run():
    luigi.build([Workflow()], local_scheduler=True, workers=6, log_level="INFO")


if __name__ == "__main__":
    # new process per task
    import multiprocessing as mp

    mp.set_start_method("spawn")
    app()
