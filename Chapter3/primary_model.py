"""
Primary model: moving-average crossover.
Outputs a side (+1 long, -1 short, 0 undefined) at every bar.
"""

import numpy as np
import pandas as pd


def ma_crossover_side(close: pd.Series,
                      fast: int = 10,
                      slow: int = 50) -> pd.Series:
    """
    Side = +1 if fast MA > slow MA, -1 if fast MA < slow MA, 0 otherwise.
    Returns 0 during the warm-up period before slow MA is defined.
    """
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    side = pd.Series(0, index=close.index, dtype=int)
    side[fast_ma > slow_ma] =  1
    side[fast_ma < slow_ma] = -1
    return side
