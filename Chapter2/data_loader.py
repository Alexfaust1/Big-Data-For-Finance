"""
Data loader for de Prado chapter 2 exercises.

Tick data pulled from yfinance:

    Each DataFrame has a DatetimeIndex and columns ['price', 'volume'].

"""

import numpy as np
import pandas as pd
import yfinance as yf

def load_spy_yahoo(period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """5-min bars as pseudo-ticks. Yahoo: 5m -> 60d, 1m -> 7d."""
    raw = yf.download("SPY", period=period, interval=interval,
                      progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = pd.DataFrame({
        "price": raw["Close"].astype(float),
        "volume": raw["Volume"].astype(float),
    }).dropna()
    return df[df["volume"] > 0]



if __name__ == "__main__":
    df = load_spy_yahoo()
    print(f"Ticks: {len(df):,}")
    print(f"Date range: {df.index.min()} -> {df.index.max()}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    daily = df.groupby(df.index.date).size()
    print(f"Daily tick count: min={daily.min():,}, max={daily.max():,}, "
          f"CV={daily.std()/daily.mean():.3f}")
