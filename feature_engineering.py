"""
Reusable feature engineering for trading data.

This module provides functions to add common trading features and time-based features
to OHLCV data. Designed to work for both training and live trading.
"""

import numpy as np
import pandas as pd
from typing import Optional


# Base price columns that should be preserved
BASE_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def add_time_features(df: pd.DataFrame, timestamp_col: Optional[str] = None) -> pd.DataFrame:
    """
    Add time-of-day features using sin/cos encoding for cyclical patterns.
    
    This creates features that capture:
    - Hour of day (0-23) -> sin/cos
    - Day of week (0-6, Monday=0) -> sin/cos
    - Day of month (1-31) -> sin/cos
    - Month (1-12) -> sin/cos
    
    Args:
        df: DataFrame with datetime index or timestamp column
        timestamp_col: Name of timestamp column if not using index
    
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Get datetime index
    if timestamp_col:
        if timestamp_col in df.columns:
            dt_index = pd.to_datetime(df[timestamp_col])
        else:
            raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            dt_index = df.index
        else:
            raise ValueError("DataFrame must have DatetimeIndex or provide timestamp_col")
    
    # Hour of day (0-23) - cycles every 24 hours
    hour = dt_index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week (0=Monday, 6=Sunday) - cycles every 7 days
    day_of_week = dt_index.dayofweek
    df["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Day of month (1-31) - cycles every ~30 days
    day_of_month = dt_index.day
    df["day_of_month_sin"] = np.sin(2 * np.pi * day_of_month / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * day_of_month / 31)
    
    # Month (1-12) - cycles every 12 months
    month = dt_index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    return df


def add_technical_indicators(
    df: pd.DataFrame,
    periods: list[int] = [7, 14, 21, 50, 200],
    include_rsi: bool = True,
    include_macd: bool = True,
    include_bollinger: bool = True,
    include_atr: bool = True,
) -> pd.DataFrame:
    """
    Add common technical indicators to OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        periods: List of periods for moving averages
        include_rsi: Whether to add RSI indicator
        include_macd: Whether to add MACD indicator
        include_bollinger: Whether to add Bollinger Bands
        include_atr: Whether to add Average True Range
    
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Ensure we have required columns
    required_cols = ["open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Moving Averages (SMA)
    for period in periods:
        df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    
    # Price relative to moving averages
    for period in periods:
        df[f"close_sma_{period}_ratio"] = df["close"] / df[f"sma_{period}"]
        df[f"close_ema_{period}_ratio"] = df["close"] / df[f"ema_{period}"]
    
    # RSI (Relative Strength Index)
    if include_rsi:
        df["rsi"] = calculate_rsi(df["close"], period=14)
    
    # MACD (Moving Average Convergence Divergence)
    if include_macd:
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
    
    # Bollinger Bands
    if include_bollinger:
        period = 20
        std = df["close"].rolling(window=period).std()
        df["bb_middle"] = df["close"].rolling(window=period).mean()
        df["bb_upper"] = df["bb_middle"] + (std * 2)
        df["bb_lower"] = df["bb_middle"] - (std * 2)
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_percent"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # Average True Range (ATR)
    if include_atr:
        df["atr"] = calculate_atr(df, period=14)
    
    # Price changes and returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    
    # Volatility (rolling standard deviation of returns)
    for period in [7, 14, 30]:
        df[f"volatility_{period}"] = df["returns"].rolling(window=period).std()
    
    # High-Low spread
    df["hl_spread"] = df["high"] - df["low"]
    df["hl_spread_pct"] = df["hl_spread"] / df["close"]
    
    # Price position within the day's range
    df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
    
    # Volume features
    if "volume" in df.columns:
        for period in [7, 14, 30]:
            df[f"volume_sma_{period}"] = df["volume"].rolling(window=period).mean()
            df[f"volume_ratio_{period}"] = df["volume"] / (df[f"volume_sma_{period}"] + 1e-8)
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of closing prices
        period: RSI period (default: 14)
    
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with high, low, close columns
        period: ATR period (default: 14)
    
    Returns:
        Series of ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def engineer_features(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    add_time_features_flag: bool = True,
    add_technical_indicators_flag: bool = True,
    periods: list[int] = [7, 14, 21, 50, 200],
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Main function to add all features to a DataFrame.
    
    This is the primary function to use for feature engineering.
    It combines time features and technical indicators.
    
    Args:
        df: DataFrame with OHLCV data and datetime index or timestamp column
        timestamp_col: Name of timestamp column if not using index
        add_time_features_flag: Whether to add time-of-day features
        add_technical_indicators_flag: Whether to add technical indicators
        periods: Periods for moving averages
        drop_na: Whether to drop rows with NaN values (from indicators)
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Add time features
    if add_time_features_flag:
        df = add_time_features(df, timestamp_col=timestamp_col)
    
    # Add technical indicators
    if add_technical_indicators_flag:
        df = add_technical_indicators(df, periods=periods)
    
    # Drop NaN values (from rolling calculations)
    if drop_na:
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with NaN values after feature engineering")
    
    return df


def add_interconnectivity_features(
    symbol_dfs: dict[str, pd.DataFrame],
    periods: list[int] = [7, 14, 30],
) -> dict[str, pd.DataFrame]:
    """
    Add features that express interconnectivity between multiple symbols.
    
    This function adds cross-symbol features like:
    - Price ratios (BTC/ETH, BTC/LTC, ETH/LTC)
    - Relative returns
    - Correlation features
    - Spread features
    - Relative strength
    
    Args:
        symbol_dfs: Dictionary mapping symbol names to DataFrames with 'close' column
        periods: Periods for rolling calculations
    
    Returns:
        Dictionary with updated DataFrames containing interconnectivity features
    """
    symbol_dfs = {k: v.copy() for k, v in symbol_dfs.items()}
    symbols = list(symbol_dfs.keys())
    
    if len(symbols) < 2:
        return symbol_dfs
    
    # Align all DataFrames by timestamp (inner join to keep only common timestamps)
    aligned_dfs = {}
    for symbol in symbols:
        aligned_dfs[symbol] = symbol_dfs[symbol][["close"]].copy()
    
    # Merge all closes into one DataFrame
    all_closes = pd.DataFrame(index=aligned_dfs[symbols[0]].index)
    for symbol in symbols:
        all_closes[symbol] = aligned_dfs[symbol]["close"]
    
    # Remove rows where any symbol has NaN
    all_closes = all_closes.dropna()
    
    if len(all_closes) == 0:
        print("Warning: No overlapping timestamps found for interconnectivity features")
        return symbol_dfs
    
    # Calculate price ratios between all pairs
    for i, symbol1 in enumerate(symbols):
        for symbol2 in symbols[i+1:]:
            ratio = all_closes[symbol1] / (all_closes[symbol2] + 1e-8)
            ratio_name = f"{symbol1.replace('/', '_')}_to_{symbol2.replace('/', '_')}_ratio"
            
            # Add ratio to both symbols
            for symbol in [symbol1, symbol2]:
                if ratio_name not in symbol_dfs[symbol].columns:
                    symbol_dfs[symbol][ratio_name] = ratio.reindex(symbol_dfs[symbol].index)
            
            # Add ratio returns (rate of change)
            ratio_returns = ratio.pct_change()
            ratio_returns_name = f"{symbol1.replace('/', '_')}_to_{symbol2.replace('/', '_')}_ratio_returns"
            for symbol in [symbol1, symbol2]:
                if ratio_returns_name not in symbol_dfs[symbol].columns:
                    symbol_dfs[symbol][ratio_returns_name] = ratio_returns.reindex(symbol_dfs[symbol].index)
    
    # Calculate relative returns (how each symbol performs vs others)
    for symbol in symbols:
        other_symbols = [s for s in symbols if s != symbol]
        if other_symbols:
            # Average return of other symbols
            other_returns = pd.DataFrame()
            for other in other_symbols:
                other_returns[other] = aligned_dfs[other]["close"].pct_change()
            avg_other_returns = other_returns.mean(axis=1)
            
            # This symbol's return
            symbol_returns = aligned_dfs[symbol]["close"].pct_change()
            
            # Relative return (this symbol vs average of others)
            relative_return = symbol_returns - avg_other_returns
            symbol_dfs[symbol]["relative_return_vs_others"] = relative_return.reindex(symbol_dfs[symbol].index)
    
    # Rolling correlations between symbols
    for period in periods:
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Calculate rolling correlation of returns
                returns1 = all_closes[symbol1].pct_change()
                returns2 = all_closes[symbol2].pct_change()
                
                # Align returns on common index
                returns_df = pd.DataFrame({
                    symbol1: returns1,
                    symbol2: returns2
                }).dropna()
                
                # Calculate rolling correlation
                # For 2 columns, we can use a vectorized approach
                rolling_corr = returns_df[symbol1].rolling(window=period).corr(returns_df[symbol2])
                
                corr_name = f"corr_{period}d_{symbol1.replace('/', '_')}_{symbol2.replace('/', '_')}"
                
                for symbol in [symbol1, symbol2]:
                    if corr_name not in symbol_dfs[symbol].columns:
                        symbol_dfs[symbol][corr_name] = rolling_corr.reindex(symbol_dfs[symbol].index)
    
    # Market dominance features (each symbol's share of total market cap proxy)
    # Using price as proxy (in real trading, you'd use market cap)
    total_price = all_closes.sum(axis=1)
    for symbol in symbols:
        dominance = all_closes[symbol] / (total_price + 1e-8)
        symbol_dfs[symbol]["market_dominance"] = dominance.reindex(symbol_dfs[symbol].index)
    
    # Relative strength (RSI of price ratio)
    for i, symbol1 in enumerate(symbols):
        for symbol2 in symbols[i+1:]:
            ratio = all_closes[symbol1] / (all_closes[symbol2] + 1e-8)
            ratio_rsi = calculate_rsi(ratio, period=14)
            rsi_name = f"rsi_{symbol1.replace('/', '_')}_to_{symbol2.replace('/', '_')}"
            
            for symbol in [symbol1, symbol2]:
                if rsi_name not in symbol_dfs[symbol].columns:
                    symbol_dfs[symbol][rsi_name] = ratio_rsi.reindex(symbol_dfs[symbol].index)
    
    return symbol_dfs


def prepare_for_live_trading(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    periods: list[int] = [7, 14, 21, 50, 200],
) -> pd.DataFrame:
    """
    Prepare features for live trading (keeps last row even if incomplete).
    
    This function is similar to engineer_features but doesn't drop NaN rows,
    which is useful for live trading where you want to use the most recent data
    even if some indicators haven't fully calculated yet.
    
    Args:
        df: DataFrame with OHLCV data
        timestamp_col: Name of timestamp column if not using index
        periods: Periods for moving averages
    
    Returns:
        DataFrame with engineered features (may contain NaN in last rows)
    """
    df = df.copy()
    
    # Add time features
    df = add_time_features(df, timestamp_col=timestamp_col)
    
    # Add technical indicators
    df = add_technical_indicators(df, periods=periods)
    
    # Forward fill NaN values (for live trading, use last known value)
    df = df.ffill()
    
    return df

