"""
Exercise 8.1: feature importance on a PCA-rotated dataset.

(a) Generate (X, y) with a mix of informative, redundant, and noise features.
(b) PCA-transform X -> X_dot.
(c) Compute MDI, MDA, SFI with an RF base estimator on (X_dot, y).
(d) Compare.

Code written in part with help from Claude Code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss


# -------------------------------------------------------------------------
# (a) Generate the dataset (de Prado snippet 8.7 style)
# -------------------------------------------------------------------------
def get_test_dataset(n_samples=4000, n_features=20, n_informative=8,
                     n_redundant=4, random_state=0):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant,
        shuffle=False, random_state=random_state,
    )
    # Column naming: I_* informative, R_* redundant, N_* noise
    cols = ([f"I_{i}" for i in range(n_informative)] +
            [f"R_{i}" for i in range(n_redundant)] +
            [f"N_{i}" for i in range(n_features - n_informative - n_redundant)])
    X = pd.DataFrame(X, columns=cols)
    y = pd.Series(y, name="target")
    return X, y


X, y = get_test_dataset()
print(f"(a) Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"    {sum(c.startswith('I') for c in X.columns)} informative, "
      f"{sum(c.startswith('R') for c in X.columns)} redundant, "
      f"{sum(c.startswith('N') for c in X.columns)} noise")


# -------------------------------------------------------------------------
# (b) PCA transform -> X_dot
# -------------------------------------------------------------------------
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(random_state=0)
X_dot = pd.DataFrame(pca.fit_transform(X_scaled),
                     columns=[f"PC_{i}" for i in range(X.shape[1])])
print(f"\n(b) PCA done. Explained variance ratio (first 8 PCs):")
print("    " + ", ".join(f"{v:.3f}" for v in pca.explained_variance_ratio_[:8]))


# -------------------------------------------------------------------------
# (c) Feature importance methods
# -------------------------------------------------------------------------
RF_KW = dict(n_estimators=200, max_features=1, max_depth=5,
             min_samples_leaf=20, class_weight='balanced',
             random_state=0, n_jobs=-1)


def feat_imp_mdi(X, y):
    """Mean Decrease Impurity: average of per-tree impurity-reduction importances."""
    rf = RandomForestClassifier(**RF_KW).fit(X, y)
    # Importance per tree, then averaged + normalised
    imp = pd.DataFrame([t.feature_importances_ for t in rf.estimators_],
                       columns=X.columns)
    out = pd.concat({'mean': imp.mean(), 'std': imp.std() / np.sqrt(imp.shape[0])},
                    axis=1)
    out['mean'] /= out['mean'].sum()
    return out['mean']


def feat_imp_mda(X, y, n_splits=5):
    """Mean Decrease Accuracy: permutation importance via CV (neg log-loss scoring)."""
    cv = KFold(n_splits=n_splits, shuffle=False)
    imp = pd.DataFrame(columns=X.columns, dtype=float)
    for fold, (tr, te) in enumerate(cv.split(X)):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]
        rf = RandomForestClassifier(**RF_KW).fit(X_tr, y_tr)
        # Baseline score on the untouched test fold
        base = -log_loss(y_te, rf.predict_proba(X_te),
                         labels=rf.classes_)
        for col in X.columns:
            X_te_perm = X_te.copy()
            X_te_perm[col] = np.random.permutation(X_te_perm[col].values)
            perm = -log_loss(y_te, rf.predict_proba(X_te_perm),
                             labels=rf.classes_)
            # Importance = how much score got worse when col was shuffled
            imp.loc[fold, col] = (base - perm) / -perm
    return imp.mean()


def feat_imp_sfi(X, y, n_splits=5):
    """
    Single Feature Importance: CV score using ONLY that one feature.

    We score with neg log-loss (higher = better). To express it as an
    *importance* on the same footing as MDI/MDA, we measure each feature's
    neg-log-loss improvement over the no-skill baseline (predicting the
    class prior). Higher = the feature carries more standalone signal.
    """
    cv = KFold(n_splits=n_splits, shuffle=False)
    prior = y.mean()
    scores = {}
    for col in X.columns:
        fold_scores = []
        for tr, te in cv.split(X):
            rf = RandomForestClassifier(**RF_KW).fit(X.iloc[tr][[col]], y.iloc[tr])
            ll = -log_loss(y.iloc[te], rf.predict_proba(X.iloc[te][[col]]),
                           labels=rf.classes_)
            p = np.clip(prior, 1e-6, 1 - 1e-6)
            base_proba = np.repeat([[1 - p, p]], len(te), axis=0)
            base_ll = -log_loss(y.iloc[te], base_proba, labels=rf.classes_)
            fold_scores.append(ll - base_ll)  # improvement over no-skill
        scores[col] = np.mean(fold_scores)
    return pd.Series(scores)


print("\n(c) Computing feature importances on (X_dot, y)...")
mdi = feat_imp_mdi(X_dot, y)
mda = feat_imp_mda(X_dot, y)
sfi = feat_imp_sfi(X_dot, y)

results = pd.DataFrame({'MDI': mdi, 'MDA': mda, 'SFI': sfi})
# Rank each method (1 = most important)
ranks = results.rank(ascending=False).astype(int)
ranks.columns = [f"{c}_rank" for c in ranks.columns]

print("\nFeature importances on PCA components:")
print(pd.concat([results.round(4), ranks], axis=1).to_string())

# Rank correlation between methods (Spearman)
print("\nRank correlation between methods (Spearman):")
print(results.corr(method='spearman').round(3).to_string())


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
for ax, col in zip(axes, ['MDI', 'MDA', 'SFI']):
    s = results[col].sort_values()
    ax.barh(range(len(s)), s.values)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=8)
    ax.set_title(col)
    ax.grid(alpha=0.3, axis='x')
plt.suptitle("Feature importance on PCA components (X_dot)")
plt.tight_layout()
plt.savefig('ex_8_1.png', dpi=120)
print("\nSaved ex_8_1.png")