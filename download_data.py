"""
Module for downloading minutely OHLCV data from Alpaca.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

# Import feature engineering constants
try:
    from feature_engineering import BASE_PRICE_COLUMNS
except ImportError:
    # Fallback if feature_engineering not available
    BASE_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]

# Load environment variables from .env file
load_dotenv()

DEFAULT_SYMBOLS: tuple[str, ...] = ("LTC/USD", "BTC/USD", "ETH/USD", )


def download_symbol_data(
    client: CryptoHistoricalDataClient,
    symbol: str,
    years: int = 4,
    chunk_days: int = 30,
) -> pd.DataFrame:
    """
    Download minutely OHLCV data for a single symbol.

    Args:
        client: Authenticated Alpaca historical data client.
        symbol: Ticker symbol to download.
        years: Number of years of history to request.
        chunk_days: Size of each download window to respect API limits.

    Returns:
        DataFrame indexed by timestamp with columns open/high/low/close/volume.
    """
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=years * 365)

    print(f"\nSymbol: {symbol}")
    print(f"Downloading minutely data from {start_date.date()} to {end_date.date()} in {chunk_days}-day chunks")

    all_rows = []
    current_start = start_date
    chunk_num = 1
    empty_chunks = []
    total_chunks = 0

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        total_chunks += 1
        print(f"  Chunk {chunk_num}: {current_start.date()} -> {current_end.date()}")

        request_params = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=current_start,
            end=current_end,
        )

        try:
            bars = client.get_crypto_bars(request_params)
            symbol_bars = bars.data.get(symbol, [])

            if not symbol_bars:
                print("    No data returned for this chunk.")
                empty_chunks.append((current_start.date(), current_end.date()))
            else:
                for bar in symbol_bars:
                    all_rows.append(
                        {
                            "timestamp": bar.timestamp,
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": int(bar.volume),
                        }
                    )
                print(f"    Retrieved {len(symbol_bars)} rows")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"    Error downloading chunk: {exc}")
            print("    Skipping to next chunk...")
            empty_chunks.append((current_start.date(), current_end.date()))

        time.sleep(0.5)
        current_start = current_end
        chunk_num += 1

        if chunk_num % 10 == 0:
            print(f"    Progress: {len(all_rows)} rows downloaded so far")

    if not all_rows:
        raise ValueError(f"No data downloaded for {symbol}. Check API credentials or symbol availability.")

    df = pd.DataFrame(all_rows).set_index("timestamp").sort_index()
    print(f"  Total rows downloaded for {symbol}: {len(df)}")
    
    # Report data gaps
    if empty_chunks:
        gap_percentage = (len(empty_chunks) / total_chunks) * 100
        print(f"  ⚠️  Warning: {len(empty_chunks)} out of {total_chunks} chunks had no data ({gap_percentage:.1f}%)")
        if len(empty_chunks) <= 5:
            print(f"  Missing chunks: {empty_chunks}")
        else:
            print(f"  First missing chunk: {empty_chunks[0]}")
            print(f"  Last missing chunk: {empty_chunks[-1]}")
            print(f"  (Total {len(empty_chunks)} missing chunks)")
        if gap_percentage > 20:
            print(f"  ⚠️  Large data gap detected! Consider using a different symbol or data source.")
    
    return df


def download_multi_asset_data(
    api_key: str,
    api_secret: str,
    symbols: Iterable[str] = DEFAULT_SYMBOLS,
    years: int = 4,
    chunk_days: int = 30,
) -> Dict[str, pd.DataFrame]:
    """
    Download minutely data for multiple instruments.

    Args:
        api_key: Alpaca API key.
        api_secret: Alpaca API secret.
        symbols: Iterable of ticker symbols to download.
        years: Number of years of history to request.
        chunk_days: Size of each download window.

    Returns:
        Mapping of symbol -> DataFrame of historical data.
    """
    client = CryptoHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    datasets: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        datasets[symbol] = download_symbol_data(client, symbol, years=years, chunk_days=chunk_days)

    return datasets


def _symbol_to_filename(symbol: str) -> str:
    """
    Convert API symbol format (e.g., 'BTC/USD') to filename-safe format (e.g., 'BTCUSD').
    """
    return symbol.replace("/", "")


def save_dataframes(data: Dict[str, pd.DataFrame], output_dir: str) -> Dict[str, str]:
    """
    Save individual symbol DataFrames to CSV files.
    
    Formats the data to match Time-Series-Library requirements:
    - First column: 'date' (datetime)
    - Other columns: OHLCV features
    - Last column: 'close' (used as target by default)

    Args:
        data: Mapping of symbol -> DataFrame.
        output_dir: Directory in which to save each CSV.

    Returns:
        Mapping of symbol -> absolute file path of the saved CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    for symbol, df in data.items():
        # Reset index to make timestamp a column
        df_save = df.reset_index()
        
        # Rename timestamp column to 'date' to match repository format
        # The index name is 'timestamp' from download_symbol_data
        if 'timestamp' in df_save.columns:
            df_save = df_save.rename(columns={'timestamp': 'date'})
        elif len(df_save.columns) > 0 and df_save.columns[0] not in BASE_PRICE_COLUMNS:
            # If first column is not a price column, it's likely the timestamp
            df_save = df_save.rename(columns={df_save.columns[0]: 'date'})
        
        # Ensure 'date' is the first column and 'close' is last (as target)
        cols = [c for c in df_save.columns if c != 'date']
        if 'close' in cols:
            cols.remove('close')
            df_save = df_save[['date'] + cols + ['close']]
        else:
            df_save = df_save[['date'] + cols]
        
        # Convert symbol format for filename (BTC/USD -> BTCUSD)
        filename_symbol = _symbol_to_filename(symbol)
        file_path = os.path.join(output_dir, f"{filename_symbol}.csv")
        df_save.to_csv(file_path, index=False)
        saved_paths[symbol] = os.path.abspath(file_path)
        print(f"Saved {symbol} data to {file_path}")

    return saved_paths


if __name__ == "__main__":
    # Get API credentials from environment variables
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        print("Error: Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        exit(1)
    
    print("Downloading multi-asset minutely data from Alpaca...")
    datasets = download_multi_asset_data(
        api_key=api_key,
        api_secret=api_secret,
        symbols=DEFAULT_SYMBOLS,
        years=4,
        chunk_days=30,
    )

    dataset_output_dir = os.path.join("dataset", "trading")

    print("\nSaving individual symbol CSV files...")
    save_dataframes(datasets, dataset_output_dir)

    print("\nDone!")

