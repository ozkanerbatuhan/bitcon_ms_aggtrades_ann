import polars as pl
import os
import gc
from datetime import datetime, timedelta
from utils.logger import setup_logger

# Setup Logger
logger = setup_logger("training_prep.log")

def get_month_path(base_dir, year, month):
    return os.path.join(base_dir, f"yil={year}", f"ay={month:02d}", "data.parquet")

def load_month_data(base_dir, year, month):
    path = get_month_path(base_dir, year, month)
    if os.path.exists(path):
        try:
            return pl.read_parquet(path)
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None
    return None

def calculate_rsi(expr, period=14):
    """
    Calculates RSI using Polars expressions.
    """
    delta = expr.diff()
    up = delta.clip(lower_bound=0)
    down = delta.clip(upper_bound=0).abs()
    
    # Exponential Weighted Moving Average
    # com = period - 1? No, usually span=period or com=(period-1)
    # Pandas ewm(alpha=1/period) is standard for RSI wilder smoothing?
    # Wilder's Smoothing: alpha = 1/n
    # Polars ewm_mean(alpha=...)
    alpha = 1.0 / period
    
    avg_gain = up.ewm_mean(alpha=alpha, adjust=False, min_periods=period)
    avg_loss = down.ewm_mean(alpha=alpha, adjust=False, min_periods=period)
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def prepare_month(year, month, input_base_dir, output_base_dir):
    """
    Prepares training data for a specific month using buffered loading.
    """
    logger.info(f"Preparing Training Data for {year}-{month:02d}...")
    
    # 1. Buffered Loading
    # Load Current Month
    df_curr = load_month_data(input_base_dir, year, month)
    if df_curr is None:
        logger.warning(f"Skipping {year}-{month:02d}: No data found.")
        return

    # Load Prev Month (Tail) - Need enough for Volatility (100) + Lags (10) = 110+
    # Let's take 1000 to be safe.
    prev_date = datetime(year, month, 1) - timedelta(days=1)
    df_prev = load_month_data(input_base_dir, prev_date.year, prev_date.month)
    if df_prev is not None:
        df_prev = df_prev.tail(1000)
    
    # Load Next Month (Head) - Need enough for Lookahead (50)
    # Let's take 1000.
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1
    
    df_next = load_month_data(input_base_dir, next_year, next_month)
    if df_next is not None:
        df_next = df_next.head(1000)
        
    # Concatenate with buffer
    # Note: We need to handle cases where prev/next is None
    dfs = []
    if df_prev is not None: dfs.append(df_prev)
    dfs.append(df_curr)
    if df_next is not None: dfs.append(df_next)
    
    df = pl.concat(dfs)
    
    # 2. Feature Engineering
    # Parameters
    RSI_PERIOD = 14
    VOL_WINDOW = 100
    LABEL_THRESHOLD = 0.0005 # 0.05%
    LOOKAHEAD = 50
    
    # Log Return: ln(price / price_prev)
    # OFI: delta_volume / volume
    # RSI: 14
    # Volatility: Rolling mean of log_return^2
    
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return"),
        
        pl.when(pl.col("volume") > 0)
        .then(pl.col("delta_volume") / pl.col("volume"))
        .otherwise(0.0)
        .alias("ofi"),
        
        calculate_rsi(pl.col("close"), RSI_PERIOD).alias("rsi")
    ])
    
    # Volatility depends on log_return, so separate step
    df = df.with_columns(
        (pl.col("log_return") ** 2).rolling_mean(window_size=VOL_WINDOW).alias("volatility")
    )
    
    # 3. Normalization
    # RSI -> RSI / 100
    # Log Return -> * 1000
    # Volatility -> * 1000000 (It's squared, so very small) -> Actually let's just scale reasonably.
    # User said: "Sabit bir sayı ile çarpmak... örneğin 1000 ile"
    # Log return ~ 0.0001. * 1000 -> 0.1. Good.
    # Volatility ~ (0.0001)^2 = 0.00000001. * 1e8 -> 1. 
    # Let's scale Volatility by 1e6 for now.
    
    df = df.with_columns([
        (pl.col("rsi") / 100.0).alias("rsi_norm"),
        (pl.col("log_return") * 1000.0).alias("log_ret_norm"),
        (pl.col("volatility") * 1000000.0).alias("vol_norm"),
        pl.col("ofi").alias("ofi_norm") # Already -1 to 1
    ])
    
    # 4. Lagging Strategy (10 steps)
    # We want lags for: rsi_norm, log_ret_norm, vol_norm, ofi_norm
    features = ["rsi_norm", "log_ret_norm", "vol_norm", "ofi_norm"]
    lag_cols = []
    
    for i in range(10): # 0 to 9
        # Lag 0 is the current value
        # Lag 1 is shift(1)
        # Wait, user said: "Sütun 1: RSI_t ... Sütun 10: RSI_t-9"
        # So we need lags 0, 1, ..., 9.
        for feat in features:
            lag_cols.append(pl.col(feat).shift(i).alias(f"{feat}_lag_{i}"))
            
    df = df.with_columns(lag_cols)
    
    # 5. Labeling (Lookahead)
    # Future Price = shift(-50)
    # If Future > Current * (1+T) -> 1
    # If Future < Current * (1-T) -> 2
    # Else -> 0
    
    future_price = pl.col("close").shift(-LOOKAHEAD)
    current_price = pl.col("close")
    
    df = df.with_columns(
        pl.when(future_price > current_price * (1 + LABEL_THRESHOLD))
        .then(1)
        .when(future_price < current_price * (1 - LABEL_THRESHOLD))
        .then(2)
        .otherwise(0)
        .cast(pl.UInt8)
        .alias("target")
    )
    
    # 6. Cleaning & Filtering
    # Drop rows with NaNs in Features or Target
    # NaNs appear at start (due to lags/RSI) and end (due to lookahead)
    
    # Filter back to the original month range
    # We need to know the start and end of the current month
    # But simpler: We can just filter by the Year/Month columns if we had them, 
    # or by datetime range.
    
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
        
    df = df.filter(
        (pl.col("datetime") >= start_date) & 
        (pl.col("datetime") < end_date)
    )
    
    # Now drop NaNs. 
    # Note: If buffer was not enough, we might lose some rows at start/end of the month.
    # But with 1000 buffer, we should be fine (needs 110 prev, 50 next).
    
    # Select only the Input columns and Target
    # Inputs: All lag columns
    # Target: target
    input_cols = [f"{feat}_lag_{i}" for i in range(10) for feat in features]
    final_cols = ["datetime"] + input_cols + ["target"]
    
    df = df.select(final_cols).drop_nulls()
    
    # 7. Save
    output_dir = os.path.join(output_base_dir, f"yil={year}", f"ay={month:02d}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "data.parquet")
    df.write_parquet(output_file)
    
    logger.info(f"Saved {year}-{month:02d}: {df.height} rows.")
    
    del df
    gc.collect()

def main():
    input_base_dir = r"c:\Users\ozkan\Desktop\aggTrades\processed_data"
    output_base_dir = r"c:\Users\ozkan\Desktop\aggTrades\training_data"
    
    # Years to process
    years = range(2017, 2026)
    
    for year in years:
        for month in range(1, 13):
            # Check if input exists
            if not os.path.exists(get_month_path(input_base_dir, year, month)):
                continue
                
            prepare_month(year, month, input_base_dir, output_base_dir)

if __name__ == "__main__":
    main()
