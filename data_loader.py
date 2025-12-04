"""Simple data loader for Alpaca crypto data."""

import os
import time
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from feature_engineering import engineer_features

load_dotenv()


def download_crypto_data(
    symbol: str = "BTC/USD",
    days: int = 60,
    timeframe: TimeFrame = TimeFrame.Hour,
    chunk_days: int = 30,
) -> pd.DataFrame:
    """
    Download crypto OHLCV data from Alpaca with chunking support for large date ranges.
    
    Args:
        symbol: Crypto pair (e.g., "BTC/USD")
        days: Number of days of history
        timeframe: Data granularity (Hour or Minute)
        chunk_days: Size of each download window to respect API limits (default: 30 days)
    
    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        raise ValueError("Set ALPACA_API_KEY and ALPACA_API_SECRET in .env")
    
    client = CryptoHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    
    end_date = datetime.now() - timedelta(days=1)  # Use yesterday to avoid incomplete data
    start_date = end_date - timedelta(days=days)
    
    timeframe_str = "minutely" if timeframe == TimeFrame.Minute else "hourly"
    print(f"Downloading {symbol} {timeframe_str} data from {start_date.date()} to {end_date.date()}")
    
    # For large date ranges, use chunking to respect API limits
    if days > chunk_days:
        print(f"Using chunking ({chunk_days}-day chunks) for large date range...")
        all_rows = []
        current_start = start_date
        chunk_num = 1
        total_chunks = (days + chunk_days - 1) // chunk_days  # Ceiling division
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            print(f"  Chunk {chunk_num}/{total_chunks}: {current_start.date()} -> {current_end.date()}")
            
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=current_start,
                end=current_end,
            )
            
            try:
                bars = client.get_crypto_bars(request)
                symbol_bars = bars.data.get(symbol, [])
                
                if symbol_bars:
                    for bar in symbol_bars:
                        all_rows.append(
                            {
                                "timestamp": bar.timestamp,
                                "open": float(bar.open),
                                "high": float(bar.high),
                                "low": float(bar.low),
                                "close": float(bar.close),
                                "volume": float(bar.volume),
                            }
                        )
                    print(f"    Retrieved {len(symbol_bars)} rows")
                else:
                    print(f"    No data for this chunk")
                
                # Rate limiting
                time.sleep(0.5)
            except Exception as exc:
                print(f"    Error downloading chunk: {exc}")
                print("    Continuing with next chunk...")
            
            current_start = current_end
            chunk_num += 1
            
            if chunk_num % 10 == 0:
                print(f"    Progress: {len(all_rows)} rows downloaded so far")
        
        if not all_rows:
            raise ValueError(f"No data downloaded for {symbol}")
        
        df = pd.DataFrame(all_rows).set_index("timestamp").sort_index()
        print(f"Downloaded {len(df)} total bars")
    else:
        # Small date range, single request
        request = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start_date,
            end=end_date,
        )
        
        bars = client.get_crypto_bars(request)
        symbol_bars = bars.data.get(symbol, [])
        
        if not symbol_bars:
            raise ValueError(f"No data for {symbol}")
        
        rows = [
            {
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            for bar in symbol_bars
        ]
        
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        print(f"Downloaded {len(df)} bars")
    
    return df


def load_or_download(
    symbol: str = "BTC/USD",
    days: int = 60,
    cache_dir: str = "data",
    timeframe: TimeFrame = TimeFrame.Minute,
    force_download: bool = False,
    apply_feature_engineering: bool = True,
) -> pd.DataFrame:
    """
    Load cached data or download if not available.
    
    Args:
        symbol: Crypto pair (e.g., "BTC/USD")
        days: Number of days of history
        cache_dir: Directory to cache data
        timeframe: Data granularity (Hour or Minute)
        force_download: If True, force re-download even if cache exists
        apply_feature_engineering: If True, apply feature engineering to the data
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Include timeframe in filename to differentiate cached files
    timeframe_str = "minute" if timeframe == TimeFrame.Minute else "hour"
    filename = f"{symbol.replace('/', '')}_{timeframe_str}_{days}d.csv"
    filepath = os.path.join(cache_dir, filename)
    
    if os.path.exists(filepath) and not force_download:
        print(f"Loading cached data from {filepath}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Loaded {len(df)} rows from cache")
    else:
        df = download_crypto_data(symbol=symbol, days=days, timeframe=timeframe)
        df.to_csv(filepath)
        print(f"Saved to {filepath}")
    
    # Apply feature engineering if requested
    if apply_feature_engineering:
        print("Applying feature engineering...")
        df = engineer_features(df, drop_na=True)
        print(f"After feature engineering: {len(df)} rows, {len(df.columns)} features")
    
    return df


def load_all_symbols(
    days: int = 60,
    cache_dir: str = "data",
    timeframe: TimeFrame = TimeFrame.Minute,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load and combine BTC/USD, ETH/USD, and LTC/USD into a single DataFrame.
    
    Adds interconnectivity features between the three symbols.
    
    Args:
        days: Number of days of history
        cache_dir: Directory to cache data
        timeframe: Data granularity (Hour or Minute)
        force_download: If True, force re-download even if cache exists
    
    Returns:
        Combined DataFrame with all symbols, sorted by timestamp
    """
    symbols = ["BTC/USD", "ETH/USD", "LTC/USD"]
    
    print(f"\n{'='*50}")
    print(f"Loading {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"{'='*50}")
    
    # Step 1: Load each symbol with basic feature engineering
    symbol_dfs = {}
    for symbol in symbols:
        print(f"\nLoading {symbol}...")
        df = load_or_download(
            symbol=symbol,
            days=days,
            cache_dir=cache_dir,
            timeframe=timeframe,
            force_download=force_download,
            apply_feature_engineering=True,
        )
        symbol_dfs[symbol] = df
    
    # Step 2: Add interconnectivity features
    print(f"\n{'='*50}")
    print("Adding interconnectivity features...")
    print(f"{'='*50}")
    from feature_engineering import add_interconnectivity_features
    symbol_dfs = add_interconnectivity_features(symbol_dfs)
    
    # Step 3: Add symbol identifier features and combine
    print(f"\n{'='*50}")
    print("Combining symbols...")
    print(f"{'='*50}")
    
    all_dfs = []
    for symbol in symbols:
        df = symbol_dfs[symbol].copy()
        
        # Add one-hot encoded symbol features
        for s in symbols:
            df[f"symbol_{s.replace('/', '_')}"] = 1.0 if s == symbol else 0.0
        
        all_dfs.append(df)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, axis=0)
    combined_df = combined_df.sort_index()
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total features: {len(combined_df.columns)}")
    
    if len(combined_df) > 0:
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    return combined_df


if __name__ == "__main__":
    df = load_or_download("BTC/USD", days=1460, timeframe=TimeFrame.Minute)
    print(df.head())
    print(f"\nShape: {df.shape}")

