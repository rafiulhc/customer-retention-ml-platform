"""Train the chosen model on the full training set and register it in the MLflow model registry.

Every registration creates a new version of `customer-retention-model`. The `@champion` alias marks
the version that serving loads. Promotion is deliberate: the first version becomes champion
automatically, later versions only with `--promote` (from module 8, the CI model gate decides).

Run:  uv run python -m retention.models.register [--promote]
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from importlib.metadata import version as installed_version
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from retention.config import FEATURES, TARGET
from retention.data.prepare import PROCESSED_DIR
from retention.models.champion import CHAMPION, MODEL_NAME, current_champion, model_input
from retention.models.experiment import model_params
from retention.models.reasons import BACKGROUND_ARTIFACT, background_mean
from retention.models.train import FINAL_MODEL, final_model
from retention.tracking import configure, start_run

logger = logging.getLogger(__name__)

DESCRIPTION = (
    "Probability that a customer cancels in the next period, used to rank customers for retention "
    "calls (top 10% each week). Input: one row per customer with the features in the signature. "
    "Output: [P(stays), P(leaves)]. See docs/problem_framing.md and docs/model_selection.md."
)
# What the serving environment must install to load the model. Explicit pins, because MLflow's
# inference records whatever the training process happened to import (pytest, under tests).
# mlflow-skinny, not mlflow: it is what serving installs (the full server package is a dev tool here).
RUNTIME_PACKAGES = ("mlflow-skinny", "scikit-learn", "numpy", "pandas", "scipy", "skops")


@dataclass(frozen=True)
class Registered:
    version: str
    run_id: str
    promoted: bool


def pip_requirements() -> list[str]:
    """The serving dependencies, pinned to the versions the model was trained with."""
    return [f"{package}=={installed_version(package)}" for package in RUNTIME_PACKAGES]


def register(
    train: pd.DataFrame, *, promote: bool = False, first_becomes_champion: bool = True, name: str = MODEL_NAME
) -> Registered:
    """Fit the final model on `train`, log it with signature + input example, register a new version.

    `first_becomes_champion`: the manual workflow makes the very first version champion so something can
    serve. The pipeline turns it off: there, only the promotion gate moves `@champion`.
    """
    X, y = model_input(train), train[TARGET]
    model: Any = final_model().fit(X, y)

    example = X.head(5)
    with warnings.catch_warnings():
        # MLflow warns that integer columns can't hold missing values. That's intended: tenure is
        # required by the data contract (CLEAN_SCHEMA), so a request without it should be rejected.
        warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
        signature = infer_signature(example, model.predict_proba(example))

    with start_run("champion candidate", tags={"stage": "training", "model_family": FINAL_MODEL}) as run:
        mlflow.log_params({"train_rows": len(train), "features": len(FEATURES), **model_params(model)})
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=example,
            pip_requirements=pip_requirements(),
            pyfunc_predict_fn="predict_proba",  # the business ranks by probability, not by 0/1 labels
            registered_model_name=name,
        )
        # What serving needs to explain predictions without the training data (models/reasons.py)
        mlflow.log_dict({"feature_means": background_mean(model, X)}, BACKGROUND_ARTIFACT)

    client = MlflowClient()
    version = str(info.registered_model_version)
    client.update_registered_model(name, description=DESCRIPTION)
    run_tags = client.get_run(run.info.run_id).data.tags
    for key in (
        "code.git_commit",
        "code.git_dirty",
        "data.train_sha256",
        "data.train_content_sha256",
        "model_family",
    ):
        if key in run_tags:
            client.set_model_version_tag(name, version, key, run_tags[key])

    promoted = promote or (first_becomes_champion and current_champion(client, name) is None)
    if promoted:
        client.set_registered_model_alias(name, CHAMPION, version)
    return Registered(version=version, run_id=run.info.run_id, promoted=promoted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--promote", action="store_true", help=f"point @{CHAMPION} at the new version")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    configure()
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    result = register(train, promote=args.promote)

    champion = current_champion(MlflowClient())
    logger.info("Registered %s version %s (run %s)", MODEL_NAME, result.version, result.run_id)
    if result.promoted:
        logger.info("@%s -> version %s", CHAMPION, result.version)
    else:
        logger.info("@%s stays on version %s; rerun with --promote to replace it", CHAMPION, champion)
    return 0


if __name__ == "__main__":
    sys.exit(main())
