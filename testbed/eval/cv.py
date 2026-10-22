"""Cross-validated scores for the channel check (SPEC §5.4).

Models are copied from the vendored `scripts/p6_4_ceiling.py:64-69`. Pure functions of the
arrays passed in; no store is opened.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261016  # the vendored p6_4_ceiling.py seed


def models() -> dict[str, Any]:
    return {
        "logistic_l2": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=10_000)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def group_folds(
    strata: ArrayLike, groups: ArrayLike, *, folds: int, seed: int
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """(train, test) index pairs; each group is in exactly one test fold, balanced by `strata`."""
    s, g = np.asarray(strata), np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros((s.size, 1)), s, g))


def oof_scores(
    X: ArrayLike,
    y: ArrayLike,
    *,
    groups: ArrayLike,
    strata: ArrayLike,
    model: str,
    folds: int = 5,
    seed: int,
) -> NDArray[np.float64]:
    """Out-of-fold P(class 1) for every row, from a fresh model per fold.

    `y` is binary 0/1 and `strata` (e.g. label x family strings) balances the folds; a group
    never straddles train and test.
    """
    x, labels = np.asarray(X, dtype=np.float64), np.asarray(y)
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("y must be binary 0/1")
    template = models()[model]
    out = np.full(labels.size, np.nan)
    for train, test in group_folds(strata, groups, folds=folds, seed=seed):
        fitted = clone(template).fit(x[train], labels[train])
        out[test] = fitted.predict_proba(x[test])[:, list(fitted.classes_).index(1)]
    if np.isnan(out).any():
        raise RuntimeError("some rows were never in a test fold")
    return out
