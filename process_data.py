import polars as pl
import os
import glob
import gc
from datetime import datetime, timedelta
from utils.logger import setup_logger
from utils.verifier import verify_month_data

# Setup Logger
logger = setup_logger()

def normalize_timestamp(q):
    """
    Normalizes timestamp column to milliseconds.
    Detects if timestamp is in microseconds (16 digits) or milliseconds (13 digits).
    Threshold: 10^14 (approx year 5138 in ms).
    """
    return q.with_columns(
        pl.when(pl.col("timestamp") > 100_000_000_000_000) # If > 10^14, assume microseconds
        .then(pl.col("timestamp") / 1000)
        .otherwise(pl.col("timestamp"))
        .cast(pl.Int64)
        .cast(pl.Datetime("ms"))
        .alias("datetime")
    )

def get_last_processed_state(output_base_dir):
    """
    Scans the output directory to find the last processed month and its close price.
    Returns: (last_year, last_month, last_close_price)
    """
    if not os.path.exists(output_base_dir):
        return 0, 0, None
        
    # Find all year directories
    years = sorted([int(d.split("=")[1]) for d in os.listdir(output_base_dir) if d.startswith("yil=")])
    if not years:
        return 0, 0, None
        
    last_year = years[-1]
    year_path = os.path.join(output_base_dir, f"yil={last_year}")
    
    # Find all month directories in the last year
    months = sorted([int(d.split("=")[1]) for d in os.listdir(year_path) if d.startswith("ay=")])
    if not months:
        # Check previous year if this year is empty (edge case)
        return 0, 0, None # Should not happen if structure is clean
        
    last_month = months[-1]
    
    # Read the last file to get close price
    file_path = os.path.join(year_path, f"ay={last_month:02d}", "data.parquet")
    try:
        # Read only the last row and 'close' column
        # Polars scan might be faster but read_parquet with row limit is fine for one file
        # Actually, to get the last row efficiently without reading all:
        # Use scan and tail.
        df = pl.scan_parquet(file_path).select("close").tail(1).collect()
        if df.height > 0:
            last_close = df["close"][0]
            logger.info(f"Resuming from {last_year}-{last_month:02d}. Last Close: {last_close}")
            return last_year, last_month, last_close
    except Exception as e:
        logger.error(f"Error reading state from {file_path}: {e}")
    
    return 0, 0, None

def process_month(year, month, input_base_dir, output_base_dir, last_close_price):
    """
    Processes a single month of Bitcoin aggTrades data.
    Returns the last close price of the month to be used for the next month.
    """
    month_str = f"{month:02d}"
    logger.info(f"Processing {year}-{month_str}...")
    
    # Input Path: bitcoin_data/yil=YYYY/ay=MM
    input_path = os.path.join(input_base_dir, f"yil={year}", f"ay={month_str}")
    
    # Check if input directory exists
    if not os.path.exists(input_path):
        logger.warning(f"No data found for {year}-{month_str}. Skipping.")
        return last_close_price

    files_pattern = os.path.join(input_path, "**", "*.parquet")
    
    try:
        # 1. Lazy Scan
        q = pl.scan_parquet(files_pattern)
        
        # 2. Type Conversion & Feature Engineering
        
        # Normalize Timestamp (Fix for Microseconds issue)
        q = normalize_timestamp(q)
        
        q = q.with_columns([
            # Price & Quantity: Float64 -> Float32
            pl.col("price").cast(pl.Float32),
            pl.col("quantity").cast(pl.Float32),
            
            # Feature Engineering: Signed Volume
            pl.when(pl.col("is_buyer_maker"))
            .then(pl.lit(-1.0, dtype=pl.Float32))
            .otherwise(pl.lit(1.0, dtype=pl.Float32))
            .alias("direction")
        ])
        
        # Calculate Signed Volume
        q = q.with_columns(
            (pl.col("quantity") * pl.col("direction")).alias("signed_volume")
        )
        
        # 3. Resampling (100ms)
        q = q.sort("datetime")
        
        # Aggregate first
        q = q.group_by_dynamic("datetime", every="100ms", closed="left").agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
            pl.col("signed_volume").sum().alias("delta_volume"),
            pl.len().alias("count")
        ])
        
        # Collect
        df = q.collect()
        
        if df.height == 0:
            logger.warning(f"No trades found in {year}-{month_str} after processing.")
            return last_close_price

        # 4. Upsample
        df = df.upsample(time_column="datetime", every="100ms")
        
        # 5. Continuity Logic
        df = df.with_columns(pl.col("close").forward_fill())
        
        if last_close_price is not None:
            df = df.with_columns(pl.col("close").fill_null(last_close_price))
        else:
            df = df.with_columns(pl.col("close").backward_fill())
            
        df = df.with_columns([
            pl.col("open").fill_null(pl.col("close")),
            pl.col("high").fill_null(pl.col("close")),
            pl.col("low").fill_null(pl.col("close")),
            pl.col("volume").fill_null(0),
            pl.col("delta_volume").fill_null(0),
            pl.col("count").fill_null(0).cast(pl.UInt32)
        ])
        
        # 6. Save
        output_dir = os.path.join(output_base_dir, f"yil={year}", f"ay={month_str}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_file = os.path.join(output_dir, "data.parquet")
        df.write_parquet(output_file)
        logger.info(f"Saved: {output_file}")
        
        # 7. Verify
        verify_month_data(output_file, logger)
        
        # 8. Update Relay Variable
        new_last_close = df["close"][-1]
        
        del df
        gc.collect()
        
        return new_last_close

    except Exception as e:
        logger.error(f"Error processing {year}-{month_str}: {e}")
        return last_close_price

def main():
    input_base_dir = r"c:\Users\ozkan\Desktop\aggTrades\bitcoin_data"
    output_base_dir = r"c:\Users\ozkan\Desktop\aggTrades\processed_data"
    
    # Resume Logic
    last_year, last_month, last_close_price = get_last_processed_state(output_base_dir)
    
    # Years to process
    years = range(2017, 2026) # 2017 to 2025
    
    for year in years:
        for month in range(1, 13):
            # Skip if already processed
            if year < last_year:
                continue
            if year == last_year and month <= last_month:
                continue
            
            # Optimization: Skip months before data start (Aug 2017)
            if year == 2017 and month < 8:
                continue
            
            last_close_price = process_month(year, month, input_base_dir, output_base_dir, last_close_price)
            
            if last_close_price is not None:
                logger.info(f"Handover price for next month: {last_close_price}")

if __name__ == "__main__":
    main()
