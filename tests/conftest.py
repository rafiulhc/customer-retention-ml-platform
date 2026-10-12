"""Shared fixtures: an offline MLflow store and a fake data manifest, so no test needs the server."""

import json
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest

from retention import lineage, tracking

MANIFEST = {
    "created_at": "2026-10-12T05:09:21+00:00",
    "code_version": "abc1234",
    "raw": {"file": "raw.csv", "sha256": "r" * 64, "rows": 10},
    "split": {"test_size": 0.2, "seed": 42, "stratified_on": "churn"},
    "outputs": {
        "train": {"file": "train.parquet", "rows": 8, "churn_rate": 0.25, "sha256": "a" * 64},
        "test": {"file": "test.parquet", "rows": 2, "churn_rate": 0.5, "sha256": "b" * 64},
    },
}


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(MANIFEST))
    return path


@pytest.fixture
def local_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A private tracking store per test; artifacts also stay inside tmp_path."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)  # MLflow's default artifact root is relative to the working dir
    yield uri
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def tracked(local_store: str, manifest: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """MLflow configured against the local store, with runs stamped from the fake manifest."""
    monkeypatch.setattr(lineage, "MANIFEST_PATH", manifest)
    tracking.configure(env_file=None)
    return local_store
