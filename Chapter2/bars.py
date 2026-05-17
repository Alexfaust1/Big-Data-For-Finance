"""
Bar formation for Question 2.1.

All three functions take a DataFrame with columns ['price', 'volume'] indexed
by timestamp, and a numeric threshold. They emit OHLCV-style bars on the
accumulation rule corresponding to their name.

  tick_bars   : new bar every `threshold` pseudo-ticks (rows)
  volume_bars : new bar when cumulative volume >= threshold
  dollar_bars : new bar when cumulative price*volume >= threshold
  
"""

import numpy as np
import pandas as pd


def _finalize_bar(chunk: pd.DataFrame) -> dict:
    """OHLCV summary for a slice of pseudo-ticks."""
    p = chunk["price"].to_numpy()
    v = chunk["volume"].to_numpy()
    dv = p * v
    return {
        "timestamp": chunk.index[-1],
        "open":   p[0],
        "high":   p.max(),
        "low":    p.min(),
        "close":  p[-1],
        "volume": v.sum(),
        "dollar": dv.sum(),
        "vwap":   dv.sum() / v.sum() if v.sum() > 0 else np.nan,
        "n_ticks": len(chunk),
    }


def tick_bars(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Emit a bar every `threshold` rows."""
    bars = []
    n = len(df)
    for start in range(0, n, threshold):
        chunk = df.iloc[start:start + threshold]
        if len(chunk) == 0:
            break
        bars.append(_finalize_bar(chunk))
    return pd.DataFrame(bars).set_index("timestamp")


def volume_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Emit a bar each time cumulative volume crosses the threshold."""
    bars = []
    v = df["volume"].to_numpy()
    cum = 0.0
    start = 0
    for i in range(len(df)):
        cum += v[i]
        if cum >= threshold:
            bars.append(_finalize_bar(df.iloc[start:i + 1]))
            cum = 0.0
            start = i + 1
    # leftover tail is dropped (incomplete bar)
    return pd.DataFrame(bars).set_index("timestamp")


def dollar_bars(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Emit a bar each time cumulative price*volume crosses the threshold."""
    bars = []
    p = df["price"].to_numpy()
    v = df["volume"].to_numpy()
    cum = 0.0
    start = 0
    for i in range(len(df)):
        cum += p[i] * v[i]
        if cum >= threshold:
            bars.append(_finalize_bar(df.iloc[start:i + 1]))
            cum = 0.0
            start = i + 1
    return pd.DataFrame(bars).set_index("timestamp")


def choose_thresholds(df: pd.DataFrame, target_bars: int) -> dict:
    """
    Pick thresholds so each method produces ~target_bars bars over the full sample.
    De Prado's rule of thumb: aim for ~50 bars/day; we let the caller set the target.
    """
    total_ticks  = len(df)
    total_volume = df["volume"].sum()
    total_dollar = (df["price"] * df["volume"]).sum()
    return {
        "tick":   max(1, int(round(total_ticks  / target_bars))),
        "volume": total_volume / target_bars,
        "dollar": total_dollar / target_bars,
    }


if __name__ == "__main__":
    from data_loader import load_spy_yahoo
    df = load_spy_yahoo()

    # Aim for roughly 50 bars/day over 60 days -> 3000 bars
    thr = choose_thresholds(df, target_bars=3000)
    print("Thresholds:", {k: f"{v:,.0f}" for k, v in thr.items()})

    tb = tick_bars(df,   thr["tick"])
    vb = volume_bars(df, thr["volume"])
    db = dollar_bars(df, thr["dollar"])

    print(f"\nBars produced:")
    print(f"  Tick   bars: {len(tb):>5,}")
    print(f"  Volume bars: {len(vb):>5,}")
    print(f"  Dollar bars: {len(db):>5,}")

    print("\nFirst three dollar bars:")
    print(db.head(3).round(3))
