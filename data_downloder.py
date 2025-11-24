import polars as pl
import requests
import zipfile
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
DATA_DIR = Path("bitcoin_data")
ZIP_DIR = DATA_DIR / "zip"
LOG_FILE = DATA_DIR / "download.log"
MISSING_FILE = DATA_DIR / "missing.log"

# Full production date range
START_DATE = datetime(2017, 8, 17)
END_DATE = datetime(2025, 11, 21)

MAX_WORKERS = 4  # Parallel download threads


def setup_directories():
    """Create necessary directories"""
    DATA_DIR.mkdir(exist_ok=True)
    ZIP_DIR.mkdir(exist_ok=True)


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )


def get_date_range(start_date, end_date):
    """Generate list of dates between start and end"""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def download_file(date):
    """Download a single zip file for given date"""
    date_str = date.strftime("%Y-%m-%d")
    filename = f"BTCUSDT-aggTrades-{date_str}.zip"
    url = BASE_URL + filename
    zip_path = ZIP_DIR / filename
    
    # Skip if already exists
    if zip_path.exists():
        logging.info(f"Already downloaded: {filename}")
        return True, zip_path
    
    try:
        logging.info(f"Downloading: {filename}")
        response = requests.get(url, timeout=30, stream=True)
        
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Downloaded: {filename}")
            return True, zip_path
        else:
            logging.warning(f"Failed to download {filename}: Status {response.status_code}")
            log_missing(filename, f"HTTP {response.status_code}")
            return False, None
            
    except Exception as e:
        logging.error(f"Error downloading {filename}: {str(e)}")
        log_missing(filename, str(e))
        return False, None


def extract_zip(zip_path):
    """Extract zip file and return CSV path"""
    try:
        csv_path = None
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to same directory
            zip_ref.extractall(ZIP_DIR)
            # Get CSV filename
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if csv_files:
                csv_path = ZIP_DIR / csv_files[0]
        
        return csv_path
    except Exception as e:
        logging.error(f"Error extracting {zip_path}: {str(e)}")
        log_missing(zip_path.name, f"Extract error: {str(e)}")
        return None


def csv_to_parquet(csv_path, date):
    """Convert CSV to Parquet using Polars with RAM efficiency and Hive Partitioning"""
    try:
        # Create Hive-style partition path: bitcoin_data/yil=YYYY/ay=MM/gun=DD
        year_dir = DATA_DIR / f"yil={date.year}"
        month_dir = year_dir / f"ay={date.month:02d}"
        day_dir = month_dir / f"gun={date.day:02d}"
        
        # Create directories
        day_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_path = day_dir / "data.parquet"
        
        # Skip if parquet already exists
        if parquet_path.exists():
            logging.info(f"Parquet already exists: {parquet_path}")
            return True, parquet_path
        
        logging.info(f"Converting to parquet: {csv_path.name}")
        
        # Scan CSV with Polars (lazy evaluation for RAM efficiency)
        lf = pl.scan_csv(
            csv_path,
            has_header=False,
            new_columns=["agg_trade_id", "price", "quantity", "first_trade_id", 
                        "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match"]
        )
        
        # Sort by timestamp (lazy)
        lf = lf.sort("timestamp")
        
        # Stream to parquet using sink_parquet (keeps RAM usage low)
        lf.sink_parquet(parquet_path, compression="zstd")
        
        logging.info(f"Saved parquet: {parquet_path}")
        return True, parquet_path
        
    except Exception as e:
        logging.error(f"Error converting {csv_path}: {str(e)}")
        log_missing(csv_path.name, f"Convert error: {str(e)}")
        return False, None


def cleanup_files(zip_path, csv_path):
    """Delete zip and csv files after successful conversion"""
    try:
        if zip_path and zip_path.exists():
            zip_path.unlink()
            logging.info(f"Deleted zip: {zip_path.name}")
        
        if csv_path and csv_path.exists():
            csv_path.unlink()
            logging.info(f"Deleted csv: {csv_path.name}")
            
    except Exception as e:
        logging.error(f"Error cleaning up files: {str(e)}")


def log_missing(filename, reason):
    """Log missing or failed files"""
    with open(MISSING_FILE, 'a') as f:
        f.write(f"{datetime.now()} - {filename} - {reason}\n")


def process_existing_zips():
    """Process any existing zip files in the directory"""
    logging.info("Checking for existing zip files...")
    zip_files = list(ZIP_DIR.glob("*.zip"))
    
    for zip_path in zip_files:
        try:
            # Extract date from filename
            date_str = zip_path.stem.split('-')[-3:]  # Get YYYY-MM-DD
            date = datetime.strptime('-'.join(date_str), "%Y-%m-%d")
            
            logging.info(f"Processing existing zip: {zip_path.name}")
            csv_path = extract_zip(zip_path)
            
            if csv_path and csv_path.exists():
                success, _ = csv_to_parquet(csv_path, date)
                if success:
                    cleanup_files(zip_path, csv_path)
        except Exception as e:
            logging.error(f"Error processing existing zip {zip_path}: {str(e)}")


def process_date(date):
    """Process a single date: download, extract, convert, cleanup"""
    success, zip_path = download_file(date)
    
    if not success:
        return False
    
    csv_path = extract_zip(zip_path)
    if not csv_path:
        return False
    
    success, parquet_path = csv_to_parquet(csv_path, date)
    
    if success:
        cleanup_files(zip_path, csv_path)
        return True
    
    return False





def main():
    """Main function to orchestrate the download and conversion"""
    setup_directories()
    setup_logging()
    
    logging.info("="*60)
    logging.info("Starting BTCUSDT aggTrades Download & Conversion")
    logging.info(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    logging.info("="*60)
    
    # First, process any existing zip files
    process_existing_zips()
    
    # Generate date range
    dates = get_date_range(START_DATE, END_DATE)
    logging.info(f"Total dates to process: {len(dates)}")
    
    # Process dates in parallel
    successful = 0
    failed = 0
    total_dates = len(dates)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_date, date): date for date in dates}
        
        for future in as_completed(futures):
            date = futures[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logging.error(f"Error processing {date}: {str(e)}")
                failed += 1
            
            # Progress update
            completed = successful + failed
            percentage = (completed / total_dates) * 100
            logging.info(f"Progress: {completed}/{total_dates} ({percentage:.2f}%) - Success: {successful}, Failed: {failed}")
    
    logging.info("="*60)
    logging.info(f"Download & Conversion complete!")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info("="*60)
    
    logging.info("="*60)
    logging.info("All operations complete!")
    logging.info(f"Check logs at: {LOG_FILE}")
    logging.info(f"Missing files logged at: {MISSING_FILE}")
    logging.info("="*60)



if __name__ == "__main__":
    main()
