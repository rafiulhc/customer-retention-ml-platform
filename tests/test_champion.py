"""Consumers' view of the model: load @champion from the registry and verify its contract."""

import subprocess
import sys

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from retention.config import TARGET
from retention.models.champion import (
    CHAMPION,
    MODEL_NAME,
    load_champion,
    model_input,
    predict_proba,
    reproduces,
    verify,
)
from retention.models.register import register
from retention.models.train import final_model
from tests.test_train import small_dataset


def test_load_champion_returns_the_version_the_alias_points_at(tracked: str) -> None:
    df = small_dataset()
    register(df)
    register(df, promote=True)
    model, version = load_champion()
    assert str(version.version) == "2"
    assert version.tags["model_family"] == "B3 logistic regression"
    assert predict_proba(model, model_input(df)).shape == (len(df),)


def test_a_registered_champion_passes_every_check(tracked: str) -> None:
    df = small_dataset()
    register(df)
    model, _ = load_champion()
    checks = verify(model, model_input(df))
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
    assert reproduces(model, df, model_input(df)).passed


def test_verify_catches_a_model_without_a_contract(tracked: str) -> None:
    """A hand-registered model: no signature, and it predicts 0/1 labels instead of probabilities."""
    df = small_dataset()
    X = model_input(df)
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            final_model().fit(X, df[TARGET]), name="model", registered_model_name=MODEL_NAME
        )
    MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION, "1")

    checks = {c.name: c for c in verify(load_champion()[0], X)}
    failed = {name for name, c in checks.items() if not c.passed}
    assert "signature lists the features" in failed
    assert "probabilities are valid" in failed  # it predicts 0/1 labels, and verify reports it
    assert "only by the model itself" in checks["rejects a request missing a feature"].detail


def test_serving_path_does_not_import_training_libraries() -> None:
    code = (
        "import sys, retention.models.champion; "
        "print(sorted(m for m in ('xgboost', 'lightgbm', 'optuna') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
