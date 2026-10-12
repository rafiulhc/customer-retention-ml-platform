"""The API's request and response formats.

Allowed values come from the data contract (`RAW_SCHEMA`), so the API accepts exactly what training
accepted: one source of truth, no list to keep in sync by hand. Invalid requests get a 422 that names
the field and the allowed values.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from retention.config import CATEGORICAL_FEATURES
from retention.data.clean import RENAMES
from retention.data.schema import RAW_SCHEMA

MAX_BATCH = 1000


def allowed_values() -> dict[str, list[str]]:
    """Each categorical feature's allowed values, read from the raw data contract."""
    values: dict[str, list[str]] = {}
    for raw_name, column in RAW_SCHEMA.columns.items():
        for check in column.checks:
            allowed = (check.statistics or {}).get("allowed_values")
            if allowed is not None:
                values[RENAMES[raw_name]] = [str(v) for v in allowed]
    return values


def _customer_model() -> type[BaseModel]:
    allowed = allowed_values()
    fields: dict[str, Any] = {
        "customer_id": (str | None, Field(None, description="Echoed back; never used by the model")),
        "tenure": (int, Field(ge=0, le=100, description="Months as a customer")),
        "monthly_charges": (float, Field(ge=0, le=500)),
        "total_charges": (float, Field(ge=0)),
    }
    for feature in CATEGORICAL_FEATURES:
        fields[feature] = (Literal[tuple(allowed[feature])], ...)  # e.g. Literal["DSL", "Fiber optic", "No"]
    return create_model(
        "Customer",
        __config__=ConfigDict(extra="forbid"),  # unknown fields (or excluded ones, like gender) are rejected
        **fields,
    )


Customer = _customer_model()


class Batch(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)  # type: ignore[valid-type]


class Scored(BaseModel):
    customer_id: str | None
    p_leaves: float = Field(description="Probability the customer cancels (calibrated)")
    worth_calling: bool = Field(description="p_leaves is at or above the model version's call threshold")
    reasons: list[str] = Field(description="Features pushing hardest towards leaving")


class Prediction(Scored):
    model_version: str


class BatchPrediction(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Scored]


class ModelInfo(BaseModel):
    name: str
    version: str
    threshold: float
    trained_on_commit: str | None
    data_fingerprint: str | None
    evaluation_run_id: str | None
    criteria_version: str | None
    passed_success_criteria: str | None
    ship_decision: str | None
