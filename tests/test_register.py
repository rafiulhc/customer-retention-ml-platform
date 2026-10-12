"""Model registry: versions, the champion alias, and a loadable model that honours its contract."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from retention.config import FEATURES
from retention.models.register import (
    CHAMPION,
    MODEL_NAME,
    current_champion,
    model_input,
    pip_requirements,
    register,
)
from tests.test_train import small_dataset


def test_model_input_is_the_features_with_plain_string_categories() -> None:
    X = model_input(small_dataset())
    assert list(X.columns) == FEATURES
    assert not any(isinstance(dtype, pd.CategoricalDtype) for dtype in X.dtypes)


def test_first_version_becomes_champion(tracked: str) -> None:
    result = register(small_dataset())
    assert (result.version, result.promoted) == ("1", True)
    assert current_champion(MlflowClient()) == "1"


def test_later_versions_need_an_explicit_promotion(tracked: str) -> None:
    df = small_dataset()
    register(df)
    second = register(df)
    assert (second.version, second.promoted) == ("2", False)
    assert current_champion(MlflowClient()) == "1"  # retraining never silently replaces the champion

    third = register(df, promote=True)
    assert current_champion(MlflowClient()) == third.version == "3"


def test_version_carries_lineage_and_the_model_has_a_description(tracked: str) -> None:
    result = register(small_dataset())
    client = MlflowClient()
    version = client.get_model_version(MODEL_NAME, result.version)
    assert version.tags["data.train_sha256"] == "a" * 64
    assert version.tags["model_family"] == "B3 logistic regression"
    assert "retention calls" in client.get_registered_model(MODEL_NAME).description


def test_champion_loads_by_alias_and_returns_probabilities(tracked: str) -> None:
    df = small_dataset()
    register(df)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")

    assert [c.name for c in model.metadata.get_input_schema().inputs] == FEATURES
    proba = np.asarray(model.predict(model_input(df)))
    assert proba.shape == (len(df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_signature_rejects_a_request_missing_a_feature(tracked: str) -> None:
    df = small_dataset()
    register(df)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
    with pytest.raises(mlflow.exceptions.MlflowException, match="tenure"):
        model.predict(model_input(df).drop(columns=["tenure"]))


def test_requirements_are_pinned_and_exclude_test_tools() -> None:
    reqs = pip_requirements()
    assert all("==" in r for r in reqs)
    assert any(r.startswith("scikit-learn==") for r in reqs)
    assert not any(r.startswith("pytest") for r in reqs)
