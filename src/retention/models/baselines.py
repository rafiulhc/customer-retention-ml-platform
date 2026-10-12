"""The rule-based baselines from docs/problem_framing.md, as scikit-learn-style estimators.

Because they follow the same fit/predict_proba interface as real models, exactly the same
cross-validation and metric code scores them: rules and models are compared on equal terms.
Their scores rank customers; they are not calibrated probabilities.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

FloatArray = npt.NDArray[np.float64]


class _RuleBaseline(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Shared plumbing: a rule only needs `_score(X)`, returning a churn score in [0, 1] per row."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        self.classes_ = np.array([0, 1])
        self._learn(X)
        return self

    def _learn(self, X: pd.DataFrame) -> None:
        """Override if the rule needs something from the training data."""

    def _score(self, X: pd.DataFrame) -> FloatArray:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> FloatArray:
        churn = np.clip(self._score(X), 0.0, 1.0)
        return np.column_stack([1.0 - churn, churn])

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.int64]:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


class NobodyLeaves(_RuleBaseline):
    """B0: everyone scores 0. Shows why accuracy is misleading (73.5% 'accurate', useless)."""

    def _score(self, X: pd.DataFrame) -> FloatArray:
        return np.zeros(len(X))


class ContractRule(_RuleBaseline):
    """B1: flag month-to-month contracts. Many ties: it can't say which of them to contact first."""

    def _score(self, X: pd.DataFrame) -> FloatArray:
        return np.asarray(X["contract"] == "Month-to-month", dtype=np.float64)


class ContractTenureRule(_RuleBaseline):
    """B2: month-to-month customers first, and within each group the newest customers first.

    The spreadsheet rule a manager could apply: the real competitor for the model.
    Month-to-month scores lie in (0.5, 1.0], everyone else in [0.0, 0.5).
    """

    def _learn(self, X: pd.DataFrame) -> None:
        self.max_tenure_ = float(X["tenure"].max())  # learned from the training data only

    def _score(self, X: pd.DataFrame) -> FloatArray:
        newness = 1.0 - np.clip(X["tenure"].to_numpy(dtype=float) / (self.max_tenure_ + 1.0), 0.0, 1.0)
        month_to_month = np.asarray(X["contract"] == "Month-to-month", dtype=np.float64)
        return 0.5 * month_to_month + 0.5 * newness * 0.999  # keeps the two groups strictly apart
