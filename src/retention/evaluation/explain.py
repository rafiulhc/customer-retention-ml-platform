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

from retention.models.reasons import original_feature_of_each_column, top_reasons

__all__ = ["global_importance", "shap_values", "top_reasons"]


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
    per_feature = per_column.T.groupby(original_feature_of_each_column(pipeline), sort=False).sum().T
    return per_feature, float(np.ravel(explainer.expected_value)[0])


def global_importance(values: pd.DataFrame) -> pd.Series:
    """Mean |contribution| per feature: how much each feature moves scores, on average."""
    return values.abs().mean().sort_values(ascending=False)
