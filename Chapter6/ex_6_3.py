"""
Exercise 6.3: ensemble of decision trees vs random forest.

(a) Plain bagging of decision trees vs RF -- what differs.
(b) Configure a BaggingClassifier to behave like a RandomForestClassifier.

Code written in part with help from Claude Code
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# -------------------------------------------------------------------------
# A synthetic classification dataset
# -------------------------------------------------------------------------
X, y = make_classification(n_samples=4000, n_features=20, n_informative=8,
                           n_redundant=4, random_state=42)

N_EST = 200
SEED = 42


# -------------------------------------------------------------------------
# (a) Plain bagging of decision trees
#
# A BaggingClassifier with a default DecisionTreeClassifier base estimator
# bootstraps SAMPLES (rows) for each tree, but each tree still considers
# ALL features at every split. That's the difference from an RF.
# -------------------------------------------------------------------------
plain_bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=SEED),
    n_estimators=N_EST,
    bootstrap=True,          # sample rows with replacement
    random_state=SEED,
    n_jobs=-1,
)

# -------------------------------------------------------------------------
# A standard random forest for comparison
# -------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=N_EST,
    random_state=SEED,
    n_jobs=-1,
)

# -------------------------------------------------------------------------
# (b) BaggingClassifier configured to behave like an RF
#
# To reproduce RF behaviour with BaggingClassifier we must:
#  1. Use a DecisionTreeClassifier whose splitter randomly restricts the
#     candidate features at each node -> set max_features='sqrt' ON THE TREE.
#     (RF considers a random sqrt(n_features) subset at every split.)
#  2. Use splitter='random' OR keep 'best' but with max_features set --
#     sklearn's RF uses the 'best' splitter over a random feature subset,
#     so max_features on the tree is the key knob.
#  3. bootstrap=True at the bagging level (RF bootstraps samples).
#  4. bootstrap_features=False, max_features=1.0 at the BAGGING level --
#     the feature randomness must come from the TREE, not from bagging
#     drawing a fixed feature subset per whole tree.
# -------------------------------------------------------------------------
bagging_as_rf = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_features='sqrt',     # <-- the RF-defining knob: random feature subset per split
        splitter='best',         # RF picks the best split within that random subset
        random_state=SEED,
    ),
    n_estimators=N_EST,
    max_samples=1.0,             # each tree trained on a full-size bootstrap sample
    bootstrap=True,              # ... drawn with replacement (RF does this)
    max_features=1.0,            # bagging does NOT subset features per tree ...
    bootstrap_features=False,    # ... feature randomness lives in the tree instead
    random_state=SEED,
    n_jobs=-1,
)


def evaluate(name, model):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"  {name:<32} accuracy = {scores.mean():.4f} +/- {scores.std():.4f}")


print("5-fold CV accuracy:\n")
evaluate("Plain bagging of trees", plain_bagging)
evaluate("RandomForestClassifier", rf)
evaluate("Bagging configured as RF", bagging_as_rf)

# -------------------------------------------------------------------------
# Show that the feature-subset knob is what matters: a single tree's
# split behaviour
# -------------------------------------------------------------------------
print("\nBase estimator feature handling:")
print(f"  Plain bagging tree max_features:   "
      f"{DecisionTreeClassifier().max_features}  (None = all features per split)")
print(f"  RF-like bagging tree max_features: 'sqrt'  "
      f"(= {int(np.sqrt(X.shape[1]))} of {X.shape[1]} features per split)")
print(f"  RandomForestClassifier max_features (default): 'sqrt'")