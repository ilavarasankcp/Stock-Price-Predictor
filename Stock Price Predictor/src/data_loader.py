import os
import pandas as pd
import yfinance as yf


def download_stock_data(ticker: str, start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """Download historical stock data using yfinance.

    Returns a DataFrame with Date index and OHLCV columns.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.index = pd.to_datetime(df.index)
    return df


def create_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Create simple lag features and moving averages for prediction.

    The target will be next-day `Close` price.
    """
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()

    for lag in range(1, lags + 1):
        df[f"lag_close_{lag}"] = df["Close"].shift(lag)

    df["Target"] = df["Close"].shift(-1)
    df = df.dropna()
    return df
