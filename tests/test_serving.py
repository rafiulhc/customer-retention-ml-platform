"""The online API: contract, predictions, explanations, failure modes, latency and import weight."""

import statistics
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import mlflow
import numpy as np
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from retention.config import FEATURES, TARGET
from retention.evaluation.explain import shap_values
from retention.models.champion import MODEL_NAME, load_version, model_input
from retention.models.reasons import background_mean, contributions
from retention.models.register import register
from retention.serving.app import create_app
from retention.serving.schemas import MAX_BATCH, allowed_values
from retention.serving.service import ScoringService, ServiceNotReady
from tests.factories import learnable_dataset

TRAIN = learnable_dataset(300, seed=1)


def as_request(row: dict[str, Any], customer_id: str = "0001-ABCDE") -> dict[str, Any]:
    body = {f: row[f] for f in FEATURES}
    body = {k: (str(v) if not isinstance(v, int | float) else v) for k, v in body.items()}
    return {"customer_id": customer_id, **body}


CUSTOMER = as_request(model_input(TRAIN).iloc[0].to_dict())


@pytest.fixture
def registered(tracked: str) -> str:
    version = register(TRAIN).version
    MlflowClient().set_model_version_tag(
        MODEL_NAME, version, "decision_threshold", "0.28"
    )  # as an evaluation would
    return version


@pytest.fixture
def client(registered: str) -> Iterator[TestClient]:
    with TestClient(create_app(ScoringService.from_registry)) as test_client:  # runs startup (model loading)
        yield test_client


# ---------- the contract ----------
def test_allowed_values_come_from_the_data_contract() -> None:
    assert allowed_values()["contract"] == ["Month-to-month", "One year", "Two year"]
    assert "Fiber optic" in allowed_values()["internet_service"]


def test_predict_returns_risk_decision_reasons_and_version(client: TestClient) -> None:
    response = client.post("/predict", json=CUSTOMER)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 <= body["p_leaves"] <= 1
    assert body["worth_calling"] == (body["p_leaves"] >= 0.28)
    assert body["model_version"] == "1" and body["customer_id"] == "0001-ABCDE"
    assert all(" = " in reason for reason in body["reasons"])
    assert "X-Process-Time-Ms" in response.headers


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"contract": "Weekly"}, "contract"),  # not an allowed value
        ({"tenure": -1}, "tenure"),  # out of range
        ({"gender": "Female"}, "gender"),  # excluded from the model, so not accepted
    ],
)
def test_invalid_requests_are_rejected_with_the_field_named(
    client: TestClient, change: dict[str, Any], field: str
) -> None:
    response = client.post("/predict", json={**CUSTOMER, **change})
    assert response.status_code == 422
    assert field in response.text


def test_a_missing_feature_is_rejected(client: TestClient) -> None:
    body = {k: v for k, v in CUSTOMER.items() if k != "tenure"}
    assert client.post("/predict", json=body).status_code == 422


def test_batch_keeps_input_order_and_respects_its_limit(client: TestClient) -> None:
    rows = [
        as_request(r, f"{i:04d}-ABCDE") for i, r in enumerate(model_input(TRAIN).head(5).to_dict("records"))
    ]
    response = client.post("/predict/batch", json={"customers": rows})
    assert response.status_code == 200
    assert [p["customer_id"] for p in response.json()["predictions"]] == [r["customer_id"] for r in rows]
    too_many = client.post("/predict/batch", json={"customers": [CUSTOMER] * (MAX_BATCH + 1)})
    assert too_many.status_code == 422


def test_model_info_reports_the_version_and_its_lineage(client: TestClient) -> None:
    info = client.get("/model").json()
    assert info["version"] == "1" and info["threshold"] == 0.28
    assert info["trained_on_commit"]
    assert client.get("/health").json() == {"status": "ok", "model_version": "1"}


# ---------- same numbers as the registry, same explanations as SHAP ----------
def test_api_scores_match_the_registered_model(client: TestClient, registered: str) -> None:
    rows = model_input(TRAIN).head(20)
    body = {"customers": [as_request(r, f"{i:04d}-ABCDE") for i, r in enumerate(rows.to_dict("records"))]}
    api = np.array([p["p_leaves"] for p in client.post("/predict/batch", json=body).json()["predictions"]])
    registry = np.asarray(load_version(registered).predict(rows))[:, 1]
    assert np.allclose(api, registry, atol=1e-4)  # the API rounds to 4 decimals


def test_closed_form_reasons_equal_shap_values() -> None:
    from retention.models.train import final_model

    X = model_input(TRAIN)
    model = final_model().fit(X, TRAIN[TARGET])
    exact, _ = shap_values(model, X.head(30), background=X)
    closed_form = contributions(model, X.head(30), background_mean(model, X))
    assert np.allclose(exact.to_numpy(), closed_form.to_numpy())


# ---------- failure modes ----------
def test_a_never_evaluated_version_is_not_served(tracked: str) -> None:
    register(TRAIN)  # no decision_threshold tag
    with pytest.raises(ServiceNotReady, match="test evaluation"):
        ScoringService.from_registry()


def test_no_champion_means_no_service(tracked: str) -> None:
    with pytest.raises(ServiceNotReady, match="no customer-retention-model@champion"):
        ScoringService.from_registry()


def test_startup_fails_fast_without_a_servable_model(tracked: str) -> None:
    with pytest.raises(ServiceNotReady), TestClient(create_app(ScoringService.from_registry)):
        pass


# ---------- non-functional ----------
def test_single_prediction_latency_is_well_under_the_200ms_requirement(client: TestClient) -> None:
    client.post("/predict", json=CUSTOMER)  # warm-up
    timings = []
    for _ in range(50):
        start = time.perf_counter()
        assert client.post("/predict", json=CUSTOMER).status_code == 200
        timings.append((time.perf_counter() - start) * 1000)
    p95 = statistics.quantiles(timings, n=20)[-1]
    assert p95 < 200, f"p95 {p95:.1f} ms"


def test_serving_does_not_import_training_or_pipeline_libraries() -> None:
    code = (
        "import sys, retention.serving.app; "
        "print(sorted(m for m in ('xgboost', 'lightgbm', 'optuna', 'shap', 'prefect', 'matplotlib') "
        "if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_mlflow_is_reachable_only_through_the_client(tracked: str) -> None:
    assert mlflow.get_tracking_uri().startswith("sqlite")  # the tests never touch the real server
