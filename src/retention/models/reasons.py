"""Per-customer reasons for serving, without the SHAP library.

For a linear model, SHAP values (with an independent background) have an exact closed form:

    contribution of column j = coefficient_j x (value_j - background mean_j)      (in log-odds)

So serving needs only the background means (logged with each model version, a few dozen numbers),
not the training data or the `shap` package. `evaluation/explain.py` computes the same values with
`shap.LinearExplainer`, and a test checks the two agree.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from retention.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

BACKGROUND_ARTIFACT = "explainer/background.json"


def original_feature_of_each_column(pipeline: Pipeline) -> list[str]:
    """For every column the model sees, the input feature it came from (one-hot expanded)."""
    encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    owners = list(NUMERIC_FEATURES)
    for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True):
        owners += [feature] * len(categories)
    return owners


def background_mean(pipeline: Pipeline, background: pd.DataFrame) -> list[float]:
    """The 'average customer' in the model's own (encoded, scaled) columns."""
    encoded = np.asarray(pipeline.named_steps["preprocess"].transform(background), dtype=np.float64)
    return [float(v) for v in encoded.mean(axis=0)]


def contributions(pipeline: Pipeline, X: pd.DataFrame, mean: list[float]) -> pd.DataFrame:
    """Each original feature's push on the log-odds of leaving, per customer (one-hots summed)."""
    encoded = np.asarray(pipeline.named_steps["preprocess"].transform(X), dtype=np.float64)
    coefficients: npt.NDArray[np.float64] = pipeline.named_steps["model"].coef_[0]
    per_column = pd.DataFrame((encoded - np.asarray(mean)) * coefficients, index=X.index)
    return per_column.T.groupby(original_feature_of_each_column(pipeline), sort=False).sum().T


def top_reasons(values: pd.DataFrame, X: pd.DataFrame, k: int = 3) -> list[list[str]]:
    """For each customer, the k features pushing hardest towards leaving, e.g. 'contract = Month-to-month'."""
    reasons = []
    for idx, row in values.iterrows():
        pushing_up = row[row > 0].sort_values(ascending=False).head(k)
        reasons.append([f"{feature} = {X.at[idx, feature]}" for feature in pushing_up.index])
    return reasons
