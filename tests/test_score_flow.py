"""The weekly call list end to end: active customers in, a ranked, capped list with reasons out."""

from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient

from retention.config import BUSINESS
from retention.models.champion import MODEL_NAME
from retention.models.register import register
from retention.pipeline import score
from tests.factories import frame, learnable_dataset, raw_row


@pytest.fixture
def scoring(tracked: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefect_backend: None) -> Path:
    version = register(learnable_dataset(300, seed=1)).version
    MlflowClient().set_model_version_tag(MODEL_NAME, version, "decision_threshold", "0.28")

    rows = []
    for i in range(60):
        rows.append(
            raw_row(
                customerID=f"{i:04d}-ABCDE",
                Contract="Month-to-month" if i % 2 else "Two year",
                InternetService="Fiber optic" if i % 3 else "DSL",
                tenure=1 + i,
                TotalCharges=str(50.0 * (1 + i)),
                Churn="Yes" if i % 10 == 0 else "No",  # 6 customers already left
            )
        )
    raw = tmp_path / "raw.csv"
    frame(*rows).to_csv(raw, index=False)
    monkeypatch.setattr(score, "RAW_PATH", raw)
    monkeypatch.setattr(score, "download", lambda: raw)
    monkeypatch.setattr(score, "connect", lambda: None)  # `tracked` already points MLflow at the test store
    monkeypatch.setattr(score, "OUTPUT_DIR", tmp_path / "call_lists")
    return tmp_path / "call_lists"


def test_call_list_is_ranked_capped_and_explained(scoring: Path) -> None:
    summary = score.weekly_call_list(calls=10)
    assert summary["customers_scored"] == 54  # the 6 who already left are not scored
    assert summary["on_the_list"] == min(10, summary["worth_calling"])

    call_list = pd.read_csv(next(scoring.glob("call_list_*.csv")))
    assert call_list["rank"].tolist() == list(range(1, len(call_list) + 1))
    assert call_list["p_leaves"].is_monotonic_decreasing
    assert (call_list["p_leaves"] >= 0.28).all()  # nobody below the threshold, whatever the capacity
    assert call_list["reasons"].str.contains(" = ").all()
    assert set(call_list["model_version"].astype(str)) == {"1"}


def test_expected_value_uses_the_business_assumptions() -> None:
    p = pd.Series([BUSINESS.break_even_probability, 1.0])
    value = score.expected_value(p)
    assert value.iloc[0] == pytest.approx(0.0)  # break-even: a call is worth exactly nothing
    assert value.iloc[1] == pytest.approx(BUSINESS.true_positive_value)
