import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional
import argparse

def read_parquet_samples(file_path: str, n_samples: int = 5) -> pd.DataFrame:
    """Read first n samples from a parquet file using pyarrow for efficiency and low memory usage"""
    parquet = pq.ParquetFile(file_path)
    # Use iter_batches to read only n_samples rows
    batches = parquet.iter_batches(batch_size=n_samples)
    first_batch = next(batches)
    df = first_batch.to_pandas()
    return df.head(n_samples)

def get_column_stats(df: pd.DataFrame) -> dict:
    """Calculate min/max for numeric columns (excluding ID columns)"""
    stats = {}
    for col in df.columns:
        # Skip ID-like columns
        if 'id' in col.lower():
            print(f"Skipping ID-like column: {col}, dtype: {df[col].dtype}")
            continue
            
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32','uint32']:
            stats[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'dtype': str(df[col].dtype)
            }
        else:
            print(f"Skipping non-numeric column: {col} (dtype: {df[col].dtype})")
    return stats

def read_all_parquet_in_dir(directory: str, n_samples: int = 5) -> dict:
    """Read samples from all parquet files in directory"""
    parquet_dir = Path(directory)
    results = {}
    
    for parquet_file in sorted(parquet_dir.glob("*.parquet"))[:5]:
        try:
            # df = pd.read_parquet(parquet_file)
            df_sample = read_parquet_samples(parquet_file, n_samples)
            # column_stats = get_column_stats(df)
            results[parquet_file.name] = {
                'shape': df_sample.shape,
                'columns': list(df_sample.columns),
                'sample_data': df_sample
            }
            print(f"File: {parquet_file.name}")
            print(f"Shape: {df_sample.shape}")
            print(f"Columns: {df_sample.columns.tolist()}")
            # print(f"Column Stats (non-ID): {column_stats}")
            print(f"Sample:\n{df_sample}\n{'-'*50}")
        except Exception as e:
            print(f"Error reading {parquet_file.name}: {e}")
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read samples from parquet files in a directory.")
    parser.add_argument("parquet_dir", type=str, help="Directory containing parquet files")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to read from each file")
    args = parser.parse_args()

    # Read all parquet files in directory
    all_data = read_all_parquet_in_dir(args.parquet_dir, n_samples=args.n_samples)