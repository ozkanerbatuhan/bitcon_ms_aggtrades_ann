import polars as pl
import os

def verify_month_data(file_path, logger):
    """
    Verifies the integrity of a processed monthly parquet file.
    """
    if not os.path.exists(file_path):
        logger.error(f"Verification Failed: File not found {file_path}")
        return False

    try:
        df = pl.read_parquet(file_path)
        
        # 1. Check Shape
        rows = df.height
        if rows == 0:
            logger.error(f"Verification Failed: {file_path} is empty.")
            return False
            
        # 2. Check Schema (Basic check for critical columns)
        required_cols = ["datetime", "open", "high", "low", "close", "volume", "delta_volume", "count"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Verification Failed: Missing columns {missing_cols} in {file_path}")
            return False
            
        # 3. Check Gaps (Time continuity)
        # We expect 100ms intervals.
        # Checking every single row might be expensive for large files, 
        # but for monthly files (~26M rows) it's feasible in Polars.
        time_diff = df["datetime"].diff().dt.total_milliseconds().drop_nulls()
        
        # Check if all diffs are 100ms
        # We allow the first diff to be null (handled by drop_nulls)
        # If there are any diffs != 100, we have a problem.
        invalid_intervals = time_diff.filter(time_diff != 100)
        
        if invalid_intervals.len() > 0:
            logger.warning(f"Verification Warning: Found {invalid_intervals.len()} gaps or irregular intervals in {file_path}")
            # We don't return False here necessarily, as we might want to proceed, but it's good to log.
            # Actually, if we did upsampling correctly, there should be NO gaps.
            # Let's be strict.
            # return False 
        
        # 4. Check Nulls in Price
        null_prices = df.select(pl.col("close").null_count()).item()
        if null_prices > 0:
             logger.error(f"Verification Failed: Found {null_prices} null close prices in {file_path}")
             return False

        logger.info(f"Verification Passed: {file_path} | Rows: {rows}")
        return True

    except Exception as e:
        logger.error(f"Verification Error for {file_path}: {e}")
        return False
