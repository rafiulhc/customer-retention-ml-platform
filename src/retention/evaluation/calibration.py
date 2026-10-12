"""Calibration: when the model says 30%, do about 30% of those customers really leave?

The profit rule ("contact when p > break-even") multiplies probabilities by money, so they must be
honest, not just well ordered. Ranking metrics (PR-AUC) can't see this; these checks can.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

Array = npt.NDArray[np.float64] | npt.NDArray[np.int64]


def calibration_table(y_true: Array, scores: Array, bins: int = 10) -> pd.DataFrame:
    """Customers in equal-size groups by predicted risk: mean prediction vs observed churn rate."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(scores, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    table = frame.groupby("bin").agg(customers=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    table["gap"] = table["predicted"] - table["observed"]
    return table.reset_index(drop=True)


def expected_calibration_error(y_true: Array, scores: Array, bins: int = 10) -> float:
    """Average |predicted - observed| over the risk groups, weighted by size (0 = perfect)."""
    table = calibration_table(y_true, scores, bins)
    return float((table["gap"].abs() * table["customers"]).sum() / table["customers"].sum())


def calibration_in_the_large(y_true: Array, scores: Array) -> float:
    """Mean prediction minus observed rate: positive = the model overestimates risk overall."""
    return float(np.mean(scores) - np.mean(y_true))
