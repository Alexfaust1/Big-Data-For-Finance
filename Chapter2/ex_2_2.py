"""
Exercise 2.2: dollar bars vs dollar imbalance bars.

Form both bar types on the same tick series, match their bar counts so
the comparison is apples-to-apples, then compare serial correlation of
log returns at lags 1-5.

Code for this exercise written in part with Claude code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_spy_yahoo  
from bars import dollar_bars
from info_bars import dollar_imbalance_bars


# --- Form both bar types --------------------------------------------------
df = load_spy_yahoo()
print(f"Loaded {len(df):,} ticks")

# Imbalance bars first, since their count is harder to control directly
dib = dollar_imbalance_bars(df, expected_ticks_per_bar=200, ewma_span=20)
n_target = len(dib)
print(f"Dollar imbalance bars: {n_target:,}")

# Match dollar bar count to imbalance bar count for fair comparison
total_dollar = (df["price"] * df["volume"]).sum()
dollar_threshold = total_dollar / n_target
db = dollar_bars(df, dollar_threshold)
print(f"Dollar bars (matched):  {len(db):,}")


# --- Log returns ---------------------------------------------------------
ret_db  = np.log(db["close"]).diff().dropna()
ret_dib = np.log(dib["close"]).diff().dropna()


# --- Serial correlation at multiple lags ---------------------------------
def autocorrs(r, lags=range(1, 6)):
    return {lag: r.autocorr(lag) for lag in lags}

ac_db  = autocorrs(ret_db)
ac_dib = autocorrs(ret_dib)

print("\n--- Serial correlation of log returns ---")
print(f"  Lag |  Dollar bars  | Imbalance bars")
print(f"  ----+---------------+----------------")
for lag in ac_db:
    print(f"   {lag}  |   {ac_db[lag]:+.4f}      |    {ac_dib[lag]:+.4f}")

# Approximate 2-sigma noise band for an autocorr estimate from n returns
se_db  = 1 / np.sqrt(len(ret_db))
se_dib = 1 / np.sqrt(len(ret_dib))
print(f"\n  2-sigma noise band: dollar +/-{2*se_db:.3f}, imbalance +/-{2*se_dib:.3f}")

# Absolute autocorr summary (book's "greater serial correlation" question)
mean_abs_db  = np.mean([abs(v) for v in ac_db.values()])
mean_abs_dib = np.mean([abs(v) for v in ac_dib.values()])
print(f"\n  Mean |autocorr| (lags 1-5):  dollar = {mean_abs_db:.4f}, "
      f"imbalance = {mean_abs_dib:.4f}")
print(f"  Greater serial correlation: "
      f"{'dollar bars' if mean_abs_db > mean_abs_dib else 'imbalance bars'}")


# --- Plot the autocorrelation profiles -----------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
lags = list(ac_db.keys())
ax.bar([l - 0.18 for l in lags], list(ac_db.values()),  width=0.35, label="Dollar bars")
ax.bar([l + 0.18 for l in lags], list(ac_dib.values()), width=0.35, label="Imbalance bars")
ax.axhline(0, color="black", linewidth=0.5)
ax.axhline( 2 * se_db, linestyle="--", color="gray", linewidth=0.7, alpha=0.6)
ax.axhline(-2 * se_db, linestyle="--", color="gray", linewidth=0.7, alpha=0.6)
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title("Serial correlation: dollar bars vs dollar imbalance bars")
ax.set_xticks(lags)
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("ex_2_2_autocorr.png", dpi=120)
print("\nSaved ex_2_2_autocorr.png")
