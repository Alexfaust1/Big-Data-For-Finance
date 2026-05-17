"""
Fractional differencing per de Prado chapter 5.

  - frac_diff(series, d, thres):    expanding-window (snippet 5.2)
  - frac_diff_ffd(series, d, thres): fixed-width window FFD (snippet 5.3)
"""

import numpy as np
import pandas as pd


def get_weights(d: float, size: int) -> np.ndarray:
    """Weights w_k for fractional differencing of order d, length size."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1]).reshape(-1, 1)


def get_weights_ffd(d: float, thres: float) -> np.ndarray:
    """Truncate weights once |w_k| < thres -> fixed-width filter."""
    w = [1.0]
    k = 1
    while True:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < thres:
            break
        w.append(w_next)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff(series: pd.Series, d: float, thres: float = 0.01) -> pd.Series:
    """Expanding-window fracdiff (snippet 5.2)."""
    w = get_weights(d, len(series))
    w_abs = np.cumsum(np.abs(w))
    w_abs /= w_abs[-1]
    skip = int((w_abs > thres).sum())

    vals = series.values
    out = np.full(len(series), np.nan)
    for i in range(skip, len(series)):
        out[i] = np.dot(w[-(i + 1):, 0], vals[:i + 1])
    return pd.Series(out, index=series.index).dropna()


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """Fixed-width window FFD (snippet 5.3)."""
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    vals = series.values
    out = np.full(len(series), np.nan)
    for i in range(width, len(series)):
        out[i] = np.dot(w[:, 0], vals[i - width:i + 1])
    return pd.Series(out, index=series.index).dropna()
