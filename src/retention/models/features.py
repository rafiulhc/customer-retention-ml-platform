"""Preprocessing, always inside a Pipeline with the model.

Encoders and scalers learn from data (category lists, means, spreads), so they must be fitted on the
training fold only. Putting them in the same Pipeline as the model guarantees that cross-validation
refits them inside every fold: nothing leaks from the validation data.
"""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retention.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Selects exactly the configured features and encodes them.

    Any other column (customer_id, the target, excluded columns) is dropped here, so a model can
    never see it, whatever the input DataFrame contains.
    """
    numeric: Any = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            # unknown categories (e.g. a new payment method in production) become all zeros, not a crash
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_pipeline(model: Any, scale_numeric: bool) -> Pipeline:
    """Preprocessing + model as one object: fitted, saved and served together."""
    return Pipeline([("preprocess", preprocessor(scale_numeric)), ("model", model)])
