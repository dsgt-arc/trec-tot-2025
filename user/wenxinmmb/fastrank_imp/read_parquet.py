import pandas as pd
from pathlib import Path
from typing import Optional

def read_parquet_samples(file_path: str, n_samples: int = 5) -> pd.DataFrame:
    """Read first n samples from a parquet file"""
    df = pd.read_parquet(file_path)
    return df.head(n_samples)

def get_column_stats(df: pd.DataFrame) -> dict:
    """Calculate min/max for numeric columns (excluding ID columns)"""
    stats = {}
    for col in df.columns:
        # Skip ID-like columns
        if 'id' in col.lower():
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
    
    for parquet_file in parquet_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            df_sample = read_parquet_samples(parquet_file, n_samples)
            column_stats = get_column_stats(df)
            results[parquet_file.name] = {
                'shape': df_sample.shape,
                'columns': list(df_sample.columns),
                'sample_data': df_sample
            }
            print(f"File: {parquet_file.name}")
            print(f"Shape: {df_sample.shape}")
            print(f"Columns: {df_sample.columns.tolist()}")
            print(f"Column Stats (non-ID): {column_stats}")
            print(f"Sample:\n{df_sample}\n{'-'*50}")
        except Exception as e:
            print(f"Error reading {parquet_file.name}: {e}")
            
    return results

if __name__ == "__main__":
    # Placeholder directory - update with your path
    PARQUET_DIR = "/home/wenxin/project/merged-graph"
    
    # Read all parquet files in directory
    all_data = read_all_parquet_in_dir(PARQUET_DIR, n_samples=3)