"""The production model as consumers see it: load `@champion` from the registry and verify it.

Kept light on purpose: serving imports this module, so it depends on MLflow and the input contract
only, never on the training code (XGBoost, LightGBM, Optuna...).

Check the champion:  uv run python -m retention.models.champion
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel

from retention.config import FEATURES, TARGET

MODEL_NAME = "customer-retention-model"
CHAMPION = "champion"


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    """The model's input contract: the selected features, with categories as plain strings.

    Requests to the API arrive as JSON strings, so the model is trained and served on the same form.
    """
    X = df[FEATURES].copy()
    categories = X.select_dtypes("category").columns
    return X.astype(dict.fromkeys(categories, str))


def current_champion(client: MlflowClient, name: str = MODEL_NAME) -> str | None:
    """The version the champion alias points at, or None if there isn't one yet."""
    try:
        return str(client.get_model_version_by_alias(name, CHAMPION).version)
    except MlflowException:
        return None


def champion_uri(name: str = MODEL_NAME) -> str:
    return f"models:/{name}@{CHAMPION}"


def load_champion(name: str = MODEL_NAME) -> tuple[PyFuncModel, ModelVersion]:
    """The model `@champion` points at, plus its registry entry (version number and lineage tags).

    Resolving the alias once and loading that exact version means the returned model and version
    always match, even if someone moves the alias in between.
    """
    version = MlflowClient().get_model_version_by_alias(name, CHAMPION)
    return mlflow.pyfunc.load_model(f"models:/{name}/{version.version}"), version


def load_version(version: str, name: str = MODEL_NAME) -> PyFuncModel:
    """One specific registered version (e.g. a challenger that isn't champion yet)."""
    model: PyFuncModel = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    return model


def predict_proba(model: PyFuncModel, X: pd.DataFrame) -> np.ndarray:
    """P(leaves) per customer: the second column of the model's [P(stays), P(leaves)] output."""
    return np.asarray(model.predict(X), dtype=np.float64)[:, 1]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def verify(model: PyFuncModel, sample: pd.DataFrame) -> list[Check]:
    """Contract checks a consumer relies on. `sample` is in model-input form (see `model_input`)."""
    checks: list[Check] = []
    inputs = model.metadata.get_input_schema()  # type: ignore[no-untyped-call]  # untyped in MLflow
    columns = inputs.input_names() if inputs is not None else []
    checks.append(Check("signature lists the features", columns == FEATURES, f"{len(columns)} columns"))

    proba = np.asarray(model.predict(sample), dtype=np.float64)
    shape_ok = proba.shape == (len(sample), 2)
    checks.append(
        Check("one [P(stays), P(leaves)] row per customer", shape_ok, f"output shape {proba.shape}")
    )
    if shape_ok:
        valid = bool(((proba >= 0) & (proba <= 1)).all() and np.allclose(proba.sum(axis=1), 1.0))
        detail = f"P(leaves) from {proba[:, 1].min():.3f} to {proba[:, 1].max():.3f}"
    else:  # e.g. a model that returns 0/1 labels: report it, don't crash on it
        valid, detail = False, "not checked: the output is not probabilities"
    checks.append(Check("probabilities are valid", valid, detail))
    again = np.asarray(model.predict(sample), dtype=np.float64)
    checks.append(
        Check("predictions are deterministic", bool(np.array_equal(proba, again)), "same input, same output")
    )

    try:
        model.predict(sample.drop(columns=[FEATURES[0]]))
        rejected, detail = False, "accepted a request missing a feature"
    except MlflowException:
        rejected, detail = True, "rejected by the MLflow signature (a clear schema error)"
    except Exception as error:  # no signature: the model itself fails, with a less helpful error
        rejected, detail = True, f"rejected only by the model itself ({type(error).__name__})"
    checks.append(Check("rejects a request missing a feature", rejected, detail))
    return checks


def reproduces(model: PyFuncModel, train: pd.DataFrame, sample: pd.DataFrame) -> Check:
    """Retrain the final model locally on the same data: the registered one must predict the same.

    Proves the registered artifact is exactly what the code produces, not a stale or hand-made file.
    Imports the training code, so it is only used by the CLI check, never by serving.
    """
    from retention.models.train import final_model

    local = final_model().fit(model_input(train), train[TARGET])
    gap = float(np.abs(local.predict_proba(sample)[:, 1] - predict_proba(model, sample)).max())
    return Check("matches a fresh local retrain", gap < 1e-9, f"max difference {gap:.1e}")


def main() -> int:
    from retention.data.prepare import PROCESSED_DIR
    from retention.lineage import same_training_data
    from retention.tracking import configure

    configure()
    model, version = load_champion()
    print(f"{MODEL_NAME} @{CHAMPION} -> version {version.version}")
    for key in ("model_family", "code.git_commit", "code.git_dirty", "data.train_sha256"):
        print(f"  {key:<18} {version.tags.get(key, '?')}")

    train_path = PROCESSED_DIR / "train.parquet"
    train = pd.read_parquet(train_path)
    sample = model_input(train.head(200))
    checks = verify(model, sample)
    if same_training_data(version.tags, train_path):
        checks.append(reproduces(model, train, sample))
    else:  # different data: a retrain can't be compared, and the champion may be stale
        checks.append(Check("trained on the current training data", False, "train.parquet has changed"))

    for check in checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
