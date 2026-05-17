"""
Triple-barrier method (de Prado snippets 3.2, 3.3, 3.4, 3.7).

Pipeline:
  add_vertical_barriers(close, t_events, num_days) -> t1
  apply_triple_barrier(close, events, pt_sl)       -> per-barrier touch times
  get_meta_labels(close, events)                   -> binary {0, 1} labels

`events` is a DataFrame with columns ['t1', 'trgt', 'side'] indexed by event time.
"""

import numpy as np
import pandas as pd


def add_vertical_barriers(close: pd.Series,
                          t_events: pd.Index,
                          num_days: float = 1.0) -> pd.Series:
    """
    For each event time, return the timestamp of the first bar at least
    `num_days` later. NaT if past the end of the data.
    """
    target_ts = t_events + pd.Timedelta(days=num_days)
    pos = close.index.searchsorted(target_ts)
    out = pd.Series(pd.NaT, index=t_events)
    valid = pos < close.shape[0]
    out.loc[t_events[valid]] = close.index[pos[valid]]
    return out


def apply_triple_barrier(close: pd.Series,
                         events: pd.DataFrame,
                         pt_sl: list) -> pd.DataFrame:
    """
    For each event, find the first time the profit/stop barriers are touched.

    events columns: t1 (vertical barrier), trgt (vol target), side (+/-1)
    pt_sl: [profit_mult, stop_mult] in units of trgt

    Returns DataFrame with columns [t1, pt, sl] giving the timestamp each
    barrier was first touched (NaT if never).
    """
    out = events[['t1']].copy()
    out['pt'] = pd.NaT
    out['sl'] = pd.NaT

    pt_width = pt_sl[0] * events['trgt']
    sl_width = -pt_sl[1] * events['trgt']

    for loc, t1 in events['t1'].fillna(close.index[-1]).items():
        # Price path from event time through vertical barrier
        path = close.loc[loc:t1]
        # Side-adjusted cumulative return along the path
        side_adj_ret = (path / close.loc[loc] - 1) * events.at[loc, 'side']
        # First touch of each barrier (NaT if never)
        pt_touch = side_adj_ret[side_adj_ret >  pt_width.loc[loc]].index.min()
        sl_touch = side_adj_ret[side_adj_ret <  sl_width.loc[loc]].index.min()
        out.at[loc, 'pt'] = pt_touch
        out.at[loc, 'sl'] = sl_touch

    return out


def get_meta_labels(close: pd.Series,
                    events: pd.DataFrame) -> pd.DataFrame:
    """
    For each event, take the FIRST barrier touched, compute the side-adjusted
    return at that moment, and label 1 if positive, 0 otherwise.

    Required events columns: t1, pt, sl, side
    """
    # First touch = min across the three barriers
    touch_cols = ['t1', 'pt', 'sl']
    first_touch = events[touch_cols].dropna(how='all').min(axis=1)
    # Realized return at first touch, then side-adjusted
    px_end   = close.reindex(first_touch.values, method='bfill').values
    px_start = close.loc[first_touch.index].values
    ret = (px_end / px_start - 1) * events.loc[first_touch.index, 'side'].values

    out = pd.DataFrame(index=first_touch.index)
    out['first_touch'] = first_touch.values
    out['ret']         = ret
    out['bin']         = (ret > 0).astype(int)
    return out
