"""
Information-driven bars (Tick imbalance bars).

Currently:
  - apply_tick_rule: classify each tick as +1 (buy) or -1 (sell)
  - dollar_imbalance_bars: sample when cumulative signed dollar volume
    exceeds the dynamic expected-imbalance threshold
"""

import numpy as np
import pandas as pd

from bars import _finalize_bar


def apply_tick_rule(prices: np.ndarray) -> np.ndarray:
    """
    Classify each tick using the tick rule:
        b_t = +1 if p_t > p_{t-1}
        b_t = -1 if p_t < p_{t-1}
        b_t = b_{t-1} if p_t == p_{t-1}
    The very first tick is conventionally assigned +1.
    """
    out = np.empty(len(prices), dtype=np.int8)
    last = 1
    prev = prices[0]
    for i, p in enumerate(prices):
        if p > prev:
            last = 1
        elif p < prev:
            last = -1
        out[i] = last
        prev = p
    return out


def dollar_imbalance_bars(df: pd.DataFrame,
                          expected_ticks_per_bar: float = 200,
                          ewma_span: int = 20) -> pd.DataFrame:
    """
    A new bar is sampled when |theta_T| >= threshold, where:

        theta_T  = sum_{t in bar} b_t * v_t * p_t   (running signed $ volume)
        threshold = EWMA over completed bars of |theta_T_realized|

    Parameters:

    expected_ticks_per_bar : sets the threshold and controls bar count
    ewma_span : smoothing span (in bars) for the threshold update
    """
    p = df["price"].to_numpy()
    v = df["volume"].to_numpy()
    b = apply_tick_rule(p)
    signed_dv = b * v * p

    # Set initial threshold: chunk the first portion of ticks into
    # pieces of size expected_ticks_per_bar, take the median absolute imbalance.
    n_init = min(len(df), int(expected_ticks_per_bar * 30))
    chunk = int(expected_ticks_per_bar)
    init_imbalances = [
        abs(signed_dv[k:k + chunk].sum())
        for k in range(0, n_init - chunk, chunk)
    ]
    if not init_imbalances:
        raise ValueError("Not enough data for bootstrap.")
    threshold = float(np.median(init_imbalances))

    alpha = 2.0 / (ewma_span + 1)

    bars = []
    theta = 0.0
    start = 0

    for i in range(len(df)):
        theta += signed_dv[i]
        if abs(theta) >= threshold:
            bars.append(_finalize_bar(df.iloc[start:i + 1]))
            threshold = alpha * abs(theta) + (1 - alpha) * threshold
            theta = 0.0
            start = i + 1

    return pd.DataFrame(bars).set_index("timestamp")


if __name__ == "__main__":
    from data_loader import load_synthetic_ticks
    df = load_synthetic_ticks()
    print(f"Loaded {len(df):,} ticks")

    b = apply_tick_rule(df["price"].to_numpy())
    print(f"Tick rule: {(b == 1).sum():,} buys, {(b == -1).sum():,} sells, "
          f"net imbalance per tick = {b.mean():+.4f}")

    dib = dollar_imbalance_bars(df, expected_ticks_per_bar=200)
    print(f"Dollar imbalance bars: {len(dib):,}")
    print(dib.head(3).round(2))
