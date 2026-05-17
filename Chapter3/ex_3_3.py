"""
Exercise 3.3: meta-labeling on top of an MA-crossover primary model.

(a) Derive meta labels for ptSl = [1, 2], t1 with numDays = 1, trgt = daily vol.
(b) Train a random forest to decide whether to trade {0, 1}.

Code for this exercise written in part with Claude code
"""
import sys
sys.path.append(r'C:\CooperUnion\Masters\BigDataForFinance\Chapter2')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from data_loader import load_spy_yahoo   
from bars import dollar_bars, choose_thresholds
from primary_model import ma_crossover_side
from volatility import get_daily_vol
from triple_barrier import (add_vertical_barriers, apply_triple_barrier,
                            get_meta_labels)


# =========================================================================
# 1. Bars
# =========================================================================
ticks = load_spy_yahoo()
print(f"Loaded {len(ticks):,} ticks")

thr = choose_thresholds(ticks, target_bars=3000)
bars = dollar_bars(ticks, thr['dollar'])
# Strip timezone for cleaner pd.Timedelta arithmetic later
if bars.index.tz is not None:
    bars.index = bars.index.tz_localize(None)
print(f"Formed {len(bars):,} dollar bars")
close = bars['close']


# =========================================================================
# 2. Primary model: MA crossover side
# =========================================================================
side = ma_crossover_side(close, fast=10, slow=50)
print(f"Side: {(side == +1).sum()} long, {(side == -1).sum()} short, "
      f"{(side == 0).sum()} undefined")


# =========================================================================
# 3. Daily volatility target (snippet 3.1)
# =========================================================================
trgt = get_daily_vol(close, span=100)
print(f"Daily vol: mean={trgt.mean():.4f}, median={trgt.median():.4f}")


# =========================================================================
# 4. Define events & vertical barriers
# =========================================================================
# Events = every bar where both side and trgt are defined and side != 0
t_events = side[(side != 0) & trgt.notna()].index
t1 = add_vertical_barriers(close, t_events, num_days=1)
events = pd.DataFrame({
    't1':   t1,
    'trgt': trgt.loc[t_events],
    'side': side.loc[t_events],
}).dropna(subset=['t1', 'trgt'])
print(f"Events with valid t1 and trgt: {len(events):,}")


# =========================================================================
# 5. Apply triple-barrier with pt_sl = [1, 2]
# =========================================================================
barriers = apply_triple_barrier(close, events, pt_sl=[1, 2])
events = events.join(barriers[['pt', 'sl']])

# Diagnostic: which barrier ended each event
def first_barrier(row):
    times = {'pt': row['pt'], 'sl': row['sl'], 't1': row['t1']}
    times = {k: v for k, v in times.items() if pd.notna(v)}
    return min(times, key=times.get)

events['exit'] = events.apply(first_barrier, axis=1)
print("Exit barrier distribution:")
print(events['exit'].value_counts().to_string())


# =========================================================================
# 6. Meta labels (the answer to part (a))
# =========================================================================
labels = get_meta_labels(close, events)
print(f"\nMeta labels: {(labels['bin'] == 1).sum():,} (1 = take), "
      f"{(labels['bin'] == 0).sum():,} (0 = skip)")
print(f"Primary-model hit rate (raw): {labels['bin'].mean():.3f}")


# =========================================================================
# 7. Build features for the meta-model
# =========================================================================
log_ret = np.log(close).diff()
fast_ma = close.rolling(10).mean()
slow_ma = close.rolling(50).mean()

features = pd.DataFrame(index=close.index)
features['log_ret_1']  = log_ret
features['log_ret_5']  = log_ret.rolling(5).sum()
features['log_ret_20'] = log_ret.rolling(20).sum()
features['vol_20']     = log_ret.rolling(20).std()
features['vol_50']     = log_ret.rolling(50).std()
features['ma_dist']    = (fast_ma - slow_ma) / close  # normalized MA spread
features['trgt']       = trgt
features['side']       = side
# Volume-based feature: current bar volume relative to recent average
features['vol_ratio']  = bars['volume'] / bars['volume'].rolling(20).mean()

X = features.loc[labels.index].dropna()
y = labels.loc[X.index, 'bin']
print(f"\nFinal training set: {len(X):,} samples, {X.shape[1]} features")


# =========================================================================
# 8. Temporal train/test split  (no shuffling!)
# =========================================================================
split = int(0.7 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
print(f"Class balance (train): {y_train.mean():.3f}")
print(f"Class balance (test):  {y_test.mean():.3f}")


# =========================================================================
# 9. Train random forest meta-model
# =========================================================================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# Probability of class 1 (take the signal)
proba = rf.predict_proba(X_test)[:, 1]
preds_default = (proba >= 0.5).astype(int)


# =========================================================================
# 10. Evaluation
# =========================================================================
print("\n--- Meta-model metrics on test set (threshold 0.5) ---")
print(f"Accuracy : {accuracy_score(y_test, preds_default):.3f}")
print(f"Precision: {precision_score(y_test, preds_default, zero_division=0):.3f}")
print(f"Recall   : {recall_score(y_test, preds_default, zero_division=0):.3f}")
print(f"F1       : {f1_score(y_test, preds_default, zero_division=0):.3f}")
print(f"ROC AUC  : {roc_auc_score(y_test, proba):.3f}")

print("\nConfusion matrix:")
cm = confusion_matrix(y_test, preds_default)
print(pd.DataFrame(cm,
                   index=['actual_0', 'actual_1'],
                   columns=['pred_0', 'pred_1']).to_string())


# Hit rate comparison: primary alone vs primary + meta filter
print("\n--- Strategy comparison on test set ---")
primary_hit_rate = y_test.mean()
print(f"Primary alone (take every signal): {primary_hit_rate:.3f} "
      f"({len(y_test):,} trades)")

for thresh in [0.5, 0.55, 0.60, 0.65]:
    take = proba >= thresh
    n_taken = take.sum()
    if n_taken == 0:
        print(f"Meta @ {thresh}: no trades taken")
        continue
    hit = y_test[take].mean()
    print(f"Meta @ {thresh}: hit rate = {hit:.3f}  ({n_taken:,} trades, "
          f"{n_taken / len(y_test):.1%} of primary signals)")


# Feature importances
print("\n--- Feature importances ---")
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(fi.to_string())


# =========================================================================
# 11. Plot: how meta-model probability relates to outcomes
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Calibration: bin probabilities, plot actual hit rate per bin
bins_p = pd.cut(proba, np.linspace(0, 1, 11))
cal = pd.DataFrame({'p': proba, 'y': y_test.values}).groupby(bins_p, observed=True)['y'].agg(['mean', 'count'])
cal['bin_center'] = [iv.mid for iv in cal.index]
axes[0].plot([0, 1], [0, 1], '--', color='gray', label='perfect calibration')
axes[0].scatter(cal['bin_center'], cal['mean'], s=cal['count'], alpha=0.6)
axes[0].set_xlabel('Predicted probability')
axes[0].set_ylabel('Actual hit rate')
axes[0].set_title('Meta-model calibration (bubble size = N samples)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Trades-taken vs hit-rate curve as threshold sweeps
thresholds = np.linspace(0.3, 0.8, 21)
hits = []
counts = []
for t in thresholds:
    mask = proba >= t
    if mask.sum() == 0:
        hits.append(np.nan); counts.append(0)
    else:
        hits.append(y_test[mask].mean()); counts.append(mask.sum())
ax2 = axes[1]
ax2.plot(thresholds, hits, 'o-', color='C0', label='hit rate')
ax2.axhline(primary_hit_rate, linestyle='--', color='C0', alpha=0.5,
            label='primary alone')
ax2.set_xlabel('Probability threshold')
ax2.set_ylabel('Hit rate', color='C0')
ax2b = ax2.twinx()
ax2b.bar(thresholds, counts, width=0.018, alpha=0.25, color='C1')
ax2b.set_ylabel('Trades taken', color='C1')
ax2.set_title('Hit rate vs threshold (bars: trades taken)')
ax2.legend(loc='upper left')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ex_3_3_results.png', dpi=120)
print("\nSaved ex_3_3_results.png")
