"""
Exercise 2.1 from Advances in Financial Machine Learning (de Prado).

(a) Form tick, volume, and dollar bars.
(b) Weekly bar counts + plot. Which is most stable?
(c) Serial correlation of returns. Which is lowest?

Code for this exercise written in part with Claude code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_spy_yahoo   # swap for load_spy_yahoo on your machine
from bars import tick_bars, volume_bars, dollar_bars, choose_thresholds


# --- 2.1(a) Form the three bar types --------------------------------------
df = load_spy_yahoo()
print(f"Loaded {len(df):,} pseudo-ticks "
      f"({df.index.min().date()} → {df.index.max().date()})")

# Same target bars for all three -> fair comparison
TARGET = 3000
thr = choose_thresholds(df, target_bars=TARGET)
print(f"Thresholds  tick={thr['tick']}  "
      f"volume={thr['volume']:,.0f}  dollar=${thr['dollar']:,.0f}")

bars = {
    "tick":   tick_bars(df,   thr["tick"]),
    "volume": volume_bars(df, thr["volume"]),
    "dollar": dollar_bars(df, thr["dollar"]),
}
for k, b in bars.items():
    print(f"  {k:>6} bars produced: {len(b):>5,}")


# --- 2.1(b) Weekly bar counts --------------------------------------------
def weekly_counts(bar_df: pd.DataFrame) -> pd.Series:
    s = pd.Series(1, index=bar_df.index)
    # Use ISO week (year, week) so we don't bleed across year boundaries
    return s.groupby([s.index.isocalendar().year,
                      s.index.isocalendar().week]).sum()

weekly = {k: weekly_counts(b) for k, b in bars.items()}

stats = pd.DataFrame({
    "mean":      {k: w.mean()                  for k, w in weekly.items()},
    "std":       {k: w.std()                   for k, w in weekly.items()},
    "cv (std/mean)": {k: w.std() / w.mean()    for k, w in weekly.items()},
}).round(3)
print("\n--- Weekly bar count stability ---")
print(stats)
print(f"Most stable (lowest CV): {stats['cv (std/mean)'].idxmin()}")

# Plot
fig, ax = plt.subplots(figsize=(11, 4.5))
for k, w in weekly.items():
    ax.plot(range(len(w)), w.values, marker="o", label=f"{k} bars")
ax.set_xlabel("Week index")
ax.set_ylabel("Bars produced")
ax.set_title("Weekly bar count by bar type")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("weekly_counts.png", dpi=120)
print("Saved weekly_counts.png")


# --- 2.1(c) Serial correlation of returns --------------------------------
def first_order_autocorr(bar_df: pd.DataFrame) -> float:
    r = np.log(bar_df["close"]).diff().dropna()
    return r.autocorr(lag=1)

print("\n--- First-order serial correlation of log returns ---")
ac = {k: first_order_autocorr(b) for k, b in bars.items()}
for k, v in ac.items():
    print(f"  {k:>6}: {v:+.4f}")
print(f"Lowest |autocorr|: {min(ac, key=lambda k: abs(ac[k]))}")
