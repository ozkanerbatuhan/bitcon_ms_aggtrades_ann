import polars as pl
import os
import glob
import numpy as np

def check_stats():
    # Load a few files
    files = sorted(glob.glob(r"c:\Users\ozkan\Desktop\aggTrades\training_data\yil=2017\ay=*\data.parquet"))[:3]
    
    print(f"Checking {len(files)} files...")
    
    for f in files:
        print(f"\nFile: {os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}")
        df = pl.read_parquet(f)
        
        # Select feature columns (lag 0 only to save space, they are same distribution)
        cols = ["rsi_norm_lag_0", "log_ret_norm_lag_0", "vol_norm_lag_0", "ofi_norm_lag_0"]
        
        stats = df.select([
            pl.col(c).min().alias(f"{c}_min") for c in cols
        ] + [
            pl.col(c).max().alias(f"{c}_max") for c in cols
        ] + [
            pl.col(c).mean().alias(f"{c}_mean") for c in cols
        ] + [
            pl.col(c).std().alias(f"{c}_std") for c in cols
        ])
        
        # Print nicely
        print(stats.transpose(include_header=True))

if __name__ == "__main__":
    check_stats()
