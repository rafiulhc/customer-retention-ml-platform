"""Out-of-fold predictions: every training customer scored by a model that never saw them.

They behave like predictions on new data, so decisions (threshold, calibration) can be made on
them without touching the test set, which is kept for the final, one-time report.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from retention.config import CV_FOLDS, SEED, TARGET
from retention.models.champion import model_input


def out_of_fold(
    df: pd.DataFrame, make: Callable[[], Any], folds: int = CV_FOLDS, seed: int = SEED
) -> npt.NDArray[np.float64]:
    """P(leaves) for every row of `df`, each from a model trained on the other folds."""
    X, y = model_input(df), df[TARGET].to_numpy()
    scores = np.full(len(df), np.nan)
    for train_idx, valid_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        model = clone(make()).fit(X.iloc[train_idx], y[train_idx])
        scores[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    return scores
