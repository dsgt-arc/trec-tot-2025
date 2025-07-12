# /// script
# dependencies = [
#   "pyspark",
#   "typer",
# ]
# ///
from pyspark.sql import SparkSession
import typer

app = typer.Typer()


@app.command()
def to_parquet(
    input_path: str,
    output_path: str,
    num_partitions: int = 32,
    cores: int = 4,
    memory: str = "16g",
):
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .config("spark.driver.memory", memory)
        .getOrCreate()
    )
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df.repartition(num_partitions).write.parquet(output_path, mode="overwrite")
    df = spark.read.parquet(output_path)
    df.printSchema()
    df.show(5)
    spark.stop()


if __name__ == "__main__":
    app()
