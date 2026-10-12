"""Scoring, shared by the API and the weekly batch job, so both use exactly the same code.

The service loads the `@champion` version once (resolved to an exact version number), with the
call threshold chosen in its test evaluation and the background means for explanations. It refuses
to start for a version that was never evaluated: there would be no threshold to decide calls with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from retention.models.champion import CHAMPION, MODEL_NAME, model_input
from retention.models.reasons import BACKGROUND_ARTIFACT, contributions, top_reasons


class ServiceNotReady(RuntimeError):
    """The model can't be served (missing alias, threshold or explainer background)."""


@dataclass(frozen=True)
class ScoringService:
    model: Pipeline
    version: str
    threshold: float
    background: list[float]
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_registry(cls, name: str = MODEL_NAME, alias: str = CHAMPION) -> ScoringService:
        client = MlflowClient()
        try:
            version = client.get_model_version_by_alias(name, alias)
        except mlflow.exceptions.MlflowException as error:
            raise ServiceNotReady(f"no {name}@{alias} in the registry") from error
        tags = dict(version.tags)
        if "decision_threshold" not in tags:
            raise ServiceNotReady(
                f"{name} v{version.version} has no decision_threshold: run its test evaluation first"
            )
        try:
            background = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/{BACKGROUND_ARTIFACT}")
        except mlflow.exceptions.MlflowException as error:
            raise ServiceNotReady(f"{name} v{version.version} has no {BACKGROUND_ARTIFACT}") from error
        return cls(
            model=mlflow.sklearn.load_model(f"models:/{name}/{version.version}"),
            version=str(version.version),
            threshold=float(tags["decision_threshold"]),
            background=list(background["feature_means"]),
            tags=tags,
        )

    def score(self, customers: pd.DataFrame, n_reasons: int = 3) -> pd.DataFrame:
        """P(leaves), the call decision and the top reasons, one row per customer, in input order."""
        X = model_input(customers).reset_index(drop=True)
        p = self.model.predict_proba(X)[:, 1]
        reasons = top_reasons(contributions(self.model, X, self.background), X, k=n_reasons)
        return pd.DataFrame(
            {
                "p_leaves": np.round(p, 4),
                "worth_calling": p >= self.threshold,
                "reasons": reasons,
            }
        )
