"""The retraining flow end to end: a temporary Prefect backend, the offline MLflow store, small data."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from prefect.testing.utilities import prefect_test_harness

from retention import lineage
from retention.data.download import sha256_of
from retention.evaluation import evaluate, gate
from retention.lineage import content_sha256
from retention.models.champion import MODEL_NAME, current_champion
from retention.pipeline import flows
from tests.conftest import MANIFEST
from tests.factories import clean_customers
from tests.test_train import small_dataset


def learnable_dataset(n: int, seed: int) -> pd.DataFrame:
    """Churn depends on fiber internet as well as the contract: a model can beat the B2 rule (contract +
    tenure), which only knows the contract. In `small_dataset` the contract is all there is."""
    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(n):
        m2m, fiber, tenure = rng.random() < 0.5, rng.random() < 0.5, int(rng.integers(1, 72))
        churn = rng.random() < 0.03 + 0.25 * m2m + 0.55 * fiber * m2m + 0.1 * fiber
        specs.append(
            {
                "Contract": "Month-to-month" if m2m else "Two year",
                "InternetService": "Fiber optic" if fiber else "DSL",
                "tenure": tenure,
                "TotalCharges": str(50.0 * tenure),
                "Churn": "Yes" if churn else "No",
            }
        )
    return clean_customers(*specs)


WRITER = {"compression": "snappy"}  # how fake_prepare writes train.parquet; a test changes it


@pytest.fixture(scope="module")
def prefect_backend() -> Iterator[None]:
    with prefect_test_harness():  # a throwaway Prefect database, so tests never touch the real server
        yield


def _pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, train: pd.DataFrame, test: pd.DataFrame
) -> Path:
    """The flow's data steps write small files instead of downloading the real dataset."""
    data = tmp_path / "processed"
    data.mkdir()
    writer = WRITER

    def fake_prepare() -> dict[str, Any]:
        train.to_parquet(data / "train.parquet", index=False, compression=writer["compression"])
        test.to_parquet(data / "test.parquet", index=False)
        manifest = json.loads(json.dumps(MANIFEST))
        manifest["outputs"]["train"]["sha256"] = sha256_of(data / "train.parquet")
        manifest["outputs"]["train"]["content_sha256"] = content_sha256(train)
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(flows, "download", lambda: tmp_path / "raw.csv")
    monkeypatch.setattr(flows, "prepare", fake_prepare)
    monkeypatch.setattr(lineage, "MANIFEST_PATH", tmp_path / "manifest.json")
    for module in (flows, evaluate, gate):
        monkeypatch.setattr(module, "PROCESSED_DIR", data)
    monkeypatch.setattr(flows, "configure", lambda: None)  # `tracked` already configured MLflow
    return data


@pytest.fixture
def pipeline(tracked: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefect_backend: None) -> Path:
    return _pipeline(tmp_path, monkeypatch, learnable_dataset(400, seed=1), learnable_dataset(400, seed=2))


@pytest.fixture
def rule_is_enough(
    tracked: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefect_backend: None
) -> Path:
    """Data where the contract explains everything: the model can't beat the B2 rule."""
    return _pipeline(tmp_path, monkeypatch, small_dataset(120), small_dataset(80))


def test_a_first_model_that_fails_the_gate_never_becomes_champion(rule_is_enough: Path) -> None:
    outcome = flows.retrain()
    assert outcome["outcome"] == "v1 registered, not promoted"
    assert current_champion(MlflowClient()) is None  # no model beats the rule: the rule is what ships


def test_first_run_trains_evaluates_and_promotes(pipeline: Path) -> None:
    outcome = flows.retrain()
    assert outcome["outcome"] == "v1 promoted to @champion"
    assert current_champion(MlflowClient()) == "1"
    version = MlflowClient().get_model_version(MODEL_NAME, "1")
    assert version.tags["promoted_by"].startswith("pipeline gate")
    assert "evaluation_run_id" in version.tags


def test_unchanged_data_is_skipped(pipeline: Path) -> None:
    flows.retrain()
    outcome = flows.retrain()
    assert outcome["outcome"] == "skipped"
    assert "trained on exactly this data" in outcome["reason"]


def test_forced_retrain_ties_the_champion_and_replaces_it(pipeline: Path) -> None:
    flows.retrain()
    outcome = flows.retrain(force=True)
    assert outcome["outcome"] == "v2 promoted to @champion"  # same data, same model: not worse
    assert "not worse than the champion" in outcome["gate"]


def test_no_promote_mode_registers_but_keeps_the_champion(pipeline: Path) -> None:
    flows.retrain()
    outcome = flows.retrain(force=True, auto_promote=False)
    assert outcome["outcome"] == "v2 registered, not promoted"
    assert current_champion(MlflowClient()) == "1"


def test_a_broken_challenger_fails_the_run_before_evaluation(
    pipeline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from retention.models.champion import Check

    monkeypatch.setattr(flows, "verify", lambda model, sample: [Check("probabilities are valid", False, "x")])
    with pytest.raises(RuntimeError, match="failed verification"):
        flows.retrain()
    assert current_champion(MlflowClient()) is None  # registered as v1, never evaluated, never promoted
    assert not MlflowClient().get_model_version(MODEL_NAME, "1").tags.get("evaluation_run_id")


def test_same_data_in_a_differently_written_file_is_not_new_data(
    pipeline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a library upgrade rewrote identical data with different bytes, and the pipeline
    retrained. The decision now compares the data's content, not the file's checksum."""
    flows.retrain()
    before = sha256_of(pipeline / "train.parquet")
    monkeypatch.setitem(WRITER, "compression", "gzip")  # same rows, different file bytes
    outcome = flows.retrain()
    assert sha256_of(pipeline / "train.parquet") != before
    assert outcome["outcome"] == "skipped"
