"""
Exercise 5.2: ADF tests and fractional differencing on a sinusoid and its
shifted cumulative sum.

(a)     ADF on the raw sinusoid
(b.i)   ADF on the cumulative sum of the shifted sinusoid
(b.ii)  Find minimum d (expanding-window fracdiff) that yields ADF p < 0.05
(b.iii) Find minimum d (FFD)                          that yields ADF p < 0.05

Code written in part with help from Claude code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from fracdiff import frac_diff, frac_diff_ffd

# -------------------------------------------------------------------------
# Part (a)
# -------------------------------------------------------------------------
rng = np.random.default_rng(0)
N = 5000
t = np.arange(N)
# A noisy sinusoid (stationary with memory). Pure sine is too "perfectly"
# stationary; a little noise gives the ADF test a realistic problem.
sinus = pd.Series(np.sin(2 * np.pi * t / 50)
                  + 0.5 * np.sin(2 * np.pi * t / 200)
                  + 0.3 * rng.standard_normal(N),
                  index=t, name='sinus')

adf_sin = adfuller(sinus, autolag='AIC')
print("=== (a) ADF on the raw sinusoid ===")
print(f"  ADF statistic: {adf_sin[0]:.4f}")
print(f"  p-value:       {adf_sin[1]:.4g}")
print(f"  Critical values: {adf_sin[4]}")

# -------------------------------------------------------------------------
# Part (b): shift then cumulative sum
# -------------------------------------------------------------------------
shift = 0.05
shifted = sinus + shift
nonstat = shifted.cumsum().rename('nonstat')

adf_ns = adfuller(nonstat, autolag='AIC')
print("\n=== (b.i) ADF on the shifted cumulative-sum series ===")
print(f"  ADF statistic: {adf_ns[0]:.4f}")
print(f"  p-value:       {adf_ns[1]:.4g}")

# -------------------------------------------------------------------------
# (b.ii) Expanding-window fracdiff sweep
# -------------------------------------------------------------------------
print("\n=== (b.ii) Expanding-window fracdiff, thres = 1e-2 ===")
print(f"  {'d':>5} | {'ADF':>10} | {'p-value':>10} | {'len':>5}")
print(f"  ------+------------+------------+------")
ds = np.linspace(0.0, 1.0, 21)
exp_results = []
for d in ds:
    fd = frac_diff(nonstat, d=d, thres=1e-2)
    if len(fd) < 50:
        exp_results.append((d, np.nan, np.nan, len(fd)))
        print(f"  {d:5.2f} | {'--':>10} | {'--':>10} | {len(fd):>5}")
        continue
    adf = adfuller(fd, autolag='AIC')
    exp_results.append((d, adf[0], adf[1], len(fd)))
    print(f"  {d:5.2f} | {adf[0]:10.4f} | {adf[1]:10.4g} | {len(fd):>5}")

exp_min_d = next((d for d, _, p, _ in exp_results if not np.isnan(p) and p < 0.05), None)
print(f"\n  Minimum d with p < 0.05 (expanding): {exp_min_d}")

# -------------------------------------------------------------------------
# (b.iii) FFD sweep
# -------------------------------------------------------------------------
print("\n=== (b.iii) FFD, thres = 1e-5 ===")
print(f"  {'d':>5} | {'ADF':>10} | {'p-value':>10} | {'len':>5}")
print(f"  ------+------------+------------+------")
ffd_results = []
for d in ds:
    fd = frac_diff_ffd(nonstat, d=d, thres=1e-5)
    if len(fd) < 50:
        ffd_results.append((d, np.nan, np.nan, len(fd)))
        print(f"  {d:5.2f} | {'--':>10} | {'--':>10} | {len(fd):>5}")
        continue
    adf = adfuller(fd, autolag='AIC')
    ffd_results.append((d, adf[0], adf[1], len(fd)))
    print(f"  {d:5.2f} | {adf[0]:10.4f} | {adf[1]:10.4g} | {len(fd):>5}")

ffd_min_d = next((d for d, _, p, _ in ffd_results if not np.isnan(p) and p < 0.05), None)
print(f"\n  Minimum d with p < 0.05 (FFD): {ffd_min_d}")

# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

axes[0, 0].plot(sinus.iloc[:500])
axes[0, 0].set_title('(a) Sinusoid (first 500 points)')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(nonstat)
axes[0, 1].set_title('(b) cumsum(sinus + 1) — non-stationary, with memory')
axes[0, 1].grid(alpha=0.3)

exp_ds  = [d for d, _, p, _ in exp_results if not np.isnan(p)]
exp_pvs = [p for _, _, p, _ in exp_results if not np.isnan(p)]
axes[1, 0].plot(exp_ds, exp_pvs, 'o-')
axes[1, 0].axhline(0.05, linestyle='--', color='red', label='5% threshold')
axes[1, 0].set_xlabel('d'); axes[1, 0].set_ylabel('ADF p-value')
axes[1, 0].set_title('(b.ii) Expanding fracdiff: p-value vs d')
axes[1, 0].set_yscale('log')
axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

ffd_ds  = [d for d, _, p, _ in ffd_results if not np.isnan(p)]
ffd_pvs = [p for _, _, p, _ in ffd_results if not np.isnan(p)]
axes[1, 1].plot(ffd_ds, ffd_pvs, 'o-')
axes[1, 1].axhline(0.05, linestyle='--', color='red', label='5% threshold')
axes[1, 1].set_xlabel('d'); axes[1, 1].set_ylabel('ADF p-value')
axes[1, 1].set_title('(b.iii) FFD: p-value vs d')
axes[1, 1].set_yscale('log')
axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ex_5_1.png', dpi=120)
print("\nSaved ex_5_1.png")
