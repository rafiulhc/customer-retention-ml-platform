"""The whole test-set evaluation, end to end, on a small dataset and an offline MLflow store."""

import json
from pathlib import Path

import pytest
from mlflow import MlflowClient

from retention import lineage
from retention.data.download import sha256_of
from retention.evaluation import evaluate as ev
from retention.models.champion import MODEL_NAME
from retention.models.register import register
from tests.conftest import MANIFEST
from tests.test_train import small_dataset


@pytest.fixture
def evaluated_setup(tracked: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Small train/test files, a manifest with their real checksums, and a registered champion."""
    data = tmp_path / "processed"
    data.mkdir()
    train, test = small_dataset(120), small_dataset(80)
    train.to_parquet(data / "train.parquet", index=False)
    test.to_parquet(data / "test.parquet", index=False)
    manifest = json.loads(json.dumps(MANIFEST))
    manifest["outputs"]["train"]["sha256"] = sha256_of(data / "train.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(lineage, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(ev, "PROCESSED_DIR", data)
    register(train)


def test_evaluation_reports_logs_and_tags_the_model_version(evaluated_setup: None) -> None:
    report = ev.run()

    assert set(report["test"]["success_criteria"]) == {"beats_b2_on_pr_auc", "beats_b2_on_profit_at_k"}
    assert 0 <= report["decisions_on_training_oof"]["policy"]["threshold"] <= 1
    assert report["explanations"]["global_importance"]

    client = MlflowClient()
    run = client.get_run(report["mlflow_run_id"])
    assert run.data.tags["stage"] == ev.STAGE
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"evaluation.json", "segments.csv", "pr_curve.png", "shap_importance.png"} <= artifacts

    version = client.get_model_version(MODEL_NAME, "1")
    assert version.tags["evaluation_run_id"] == run.info.run_id
    assert (
        float(version.tags["decision_threshold"])
        == report["decisions_on_training_oof"]["policy"]["threshold"]
    )


def test_a_version_is_evaluated_on_the_test_set_only_once(evaluated_setup: None) -> None:
    ev.run()
    with pytest.raises(ev.AlreadyEvaluated, match="--force"):
        ev.run()
    assert ev.run(force=True)["mlflow_run_id"]


def test_refuses_a_champion_trained_on_other_data(
    evaluated_setup: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    small_dataset(90).to_parquet(ev.PROCESSED_DIR / "train.parquet", index=False)  # the data changed
    with pytest.raises(RuntimeError, match="re-register"):
        ev.run()
