"""Why the model scores a customer as it does: SHAP values, summed back to the original features.

The champion is a logistic regression, so SHAP values are exact and cheap (LinearExplainer): each
feature's contribution to the log-odds, relative to an average customer. One-hot columns
("contract = Month-to-month", "contract = Two year", ...) are added back up to one value per original
feature, which is what the customer-success team understands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from retention.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _original_feature_of_each_column(pipeline: Pipeline) -> list[str]:
    """For every column the model sees, the input feature it came from (one-hot expanded)."""
    encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    owners = list(NUMERIC_FEATURES)
    for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True):
        owners += [feature] * len(categories)
    return owners


def shap_values(pipeline: Pipeline, X: pd.DataFrame, background: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per-customer contributions to the log-odds (one column per original feature), and the base value.

    For each customer: base value + sum of contributions = the model's log-odds of leaving.
    `background` defines the "average customer" contributions are measured against (the training data).
    """
    preprocess, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    reference = preprocess.transform(background)
    # Use the whole background: shap's default silently subsamples 100 rows, a noisy "average customer".
    explainer = shap.LinearExplainer(model, shap.maskers.Independent(reference, max_samples=len(reference)))
    values = np.asarray(explainer.shap_values(preprocess.transform(X)))
    per_column = pd.DataFrame(values, index=X.index)
    per_feature = per_column.T.groupby(_original_feature_of_each_column(pipeline), sort=False).sum().T
    return per_feature, float(np.ravel(explainer.expected_value)[0])


def global_importance(values: pd.DataFrame) -> pd.Series:
    """Mean |contribution| per feature: how much each feature moves scores, on average."""
    return values.abs().mean().sort_values(ascending=False)


def top_reasons(values: pd.DataFrame, X: pd.DataFrame, k: int = 3) -> list[list[str]]:
    """For each customer, the k features pushing hardest towards leaving, e.g. 'contract = Month-to-month'."""
    reasons = []
    for idx, row in values.iterrows():
        pushing_up = row[row > 0].sort_values(ascending=False).head(k)
        reasons.append([f"{feature} = {X.at[idx, feature]}" for feature in pushing_up.index])
    return reasons
