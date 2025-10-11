import argparse
import polars as pl

def main():
    parser = argparse.ArgumentParser(description="Combine and sort parquet files by query_id")
    parser.add_argument("--input_files", nargs='+', required=True, help="List of input parquet files")
    parser.add_argument("--output_file", required=True, help="Output parquet file")
    args = parser.parse_args()

    # Read and concatenate all parquet files, casting query_id to String
    dfs = [
        pl.read_parquet(f).with_columns(
            pl.col("query_id").cast(pl.String)
        ) for f in args.input_files
    ]
    combined_df = pl.concat(dfs)

    # Sort by query_id
    sorted_df = combined_df.sort("query_id")

    # Write to output parquet file
    sorted_df.write_parquet(args.output_file)
    print(f"Combined and sorted parquet saved to {args.output_file}")

if __name__ == "__main__":
    main()