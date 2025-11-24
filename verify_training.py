import polars as pl
import os

file_path = r"c:\Users\ozkan\Desktop\aggTrades\training_data\yil=2017\ay=09\data.parquet"
df = pl.read_parquet(file_path)

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")

# Check Target Distribution
print("Target Distribution:")
print(df["target"].value_counts())

# Check Feature Ranges
print("Feature Stats:")
print(df.select([
    pl.col("rsi_norm_lag_0").min().alias("rsi_min"),
    pl.col("rsi_norm_lag_0").max().alias("rsi_max"),
    pl.col("ofi_norm_lag_0").min().alias("ofi_min"),
    pl.col("ofi_norm_lag_0").max().alias("ofi_max"),
    pl.col("log_ret_norm_lag_0").mean().alias("log_ret_mean"),
    pl.col("vol_norm_lag_0").mean().alias("vol_mean")
]))

# Check for NaNs
nulls = df.select(pl.sum_horizontal(pl.all().null_count())).item()
print(f"Total NaNs: {nulls}")
