"""
De Prado snippet 3.1: daily volatility estimator.

For each bar, find the bar from ~1 day earlier, compute the return between
them, and take an EWMA standard deviation of those daily returns. This
gives a per-bar estimate of the current daily volatility regime, which we
use as the 'trgt' for the triple-barrier method.

Fixed: replaced deprecated .iteritems() and tightened types.
"""

import numpy as np
import pandas as pd


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """
    Parameters
    ----------
    close : pd.Series of bar close prices, indexed by timestamp
    span  : EWMA span (smoothing) for the std estimator

    Returns
    -------
    pd.Series of daily-return EWMA volatility, indexed by bar timestamp.
    """
    # For each timestamp, the integer position of the bar from ~1 day earlier
    idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    idx = idx[idx > 0]
    # Map each bar to the timestamp of the previous day's bar
    prev_ts = pd.Series(close.index[idx - 1],
                        index=close.index[close.shape[0] - idx.shape[0]:])
    # Daily returns: today / day-ago - 1
    daily_ret = close.loc[prev_ts.index] / close.loc[prev_ts.values].values - 1
    # EWMA std
    return daily_ret.ewm(span=span).std()
