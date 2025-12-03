"""Simple data loader for Alpaca crypto data."""

import os
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

load_dotenv()


def download_crypto_data(
    symbol: str = "BTC/USD",
    days: int = 60,
    timeframe: TimeFrame = TimeFrame.Hour,
) -> pd.DataFrame:
    """
    Download crypto OHLCV data from Alpaca.
    
    Args:
        symbol: Crypto pair (e.g., "BTC/USD")
        days: Number of days of history
        timeframe: Data granularity (Hour recommended for quick prototyping)
    
    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        raise ValueError("Set ALPACA_API_KEY and ALPACA_API_SECRET in .env")
    
    client = CryptoHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Downloading {symbol} from {start_date.date()} to {end_date.date()}")
    
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
) -> pd.DataFrame:
    """Load cached data or download if not available."""
    os.makedirs(cache_dir, exist_ok=True)
    
    filename = symbol.replace("/", "") + ".csv"
    filepath = os.path.join(cache_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Loading cached data from {filepath}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    
    df = download_crypto_data(symbol=symbol, days=days)
    df.to_csv(filepath)
    print(f"Saved to {filepath}")
    return df


if __name__ == "__main__":
    df = load_or_download("BTC/USD", days=60)
    print(df.head())
    print(f"\nShape: {df.shape}")

