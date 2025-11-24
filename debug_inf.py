import polars as pl
import os
import numpy as np

# Check the first file of 2017
file_path = r"c:\Users\ozkan\Desktop\aggTrades\training_data\yil=2017\ay=08\data.parquet"
df = pl.read_parquet(file_path)

# Check for Infs in features
feature_cols = [c for c in df.columns if c != "target" and c != "datetime"]

print("Checking for Infs...")
for col in feature_cols:
    # Check for inf
    n_inf = df.filter(pl.col(col).is_infinite()).height
    n_nan = df.filter(pl.col(col).is_null()).height
    
    if n_inf > 0 or n_nan > 0:
        print(f"Column {col}: Inf={n_inf}, NaN={n_nan}")
        # Show sample
        print(df.filter(pl.col(col).is_infinite()).head())

print("Check complete.")
