# /// script
# dependencies = [
#   "mwsql",
#   "typer",
#   "tqdm",
# ]
# ///
import csv

import typer
from mwsql import Dump
from tqdm import tqdm

app = typer.Typer()


@app.command()
def main(input_path: str, output_path: str, mininterval: int = 10):
    try:
        dump = Dump.from_file(input_path)
    except Exception:
        dump = Dump.from_file(input_path, encoding="latin-1")
    with open(output_path, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dump.col_names)
        for row in tqdm(dump, mininterval=mininterval):
            writer.writerow(row)


if __name__ == "__main__":
    app()
