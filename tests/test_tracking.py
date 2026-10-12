"""MLflow wiring and lineage. Offline: a throwaway SQLite store in tmp_path stands in for the server."""

from pathlib import Path

import mlflow
import pytest

from retention import lineage as lin
from retention import tracking


def test_data_lineage_reads_checksums_from_the_manifest(manifest: Path) -> None:
    tags = lin.data_lineage(manifest)
    assert tags["data.raw_sha256"] == "r" * 64
    assert tags["data.train_sha256"] == "a" * 64
    assert tags["data.test_rows"] == "2"
    assert tags["data.prepared_by_commit"] == "abc1234"


def test_missing_manifest_says_how_to_fix_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"retention\.data\.prepare"):
        lin.data_lineage(tmp_path / "nope.json")


def test_lineage_includes_code_version(manifest: Path) -> None:
    tags = lin.lineage(manifest)
    assert tags["code.git_commit"]  # a hash, or "unknown" outside git
    assert tags["code.git_dirty"] in {"true", "false", "unknown"}
    assert all(isinstance(v, str) for v in tags.values())  # MLflow tags must be strings


def test_configure_uses_the_environment(local_store: str) -> None:
    assert tracking.configure(env_file=None) == local_store
    assert mlflow.get_experiment_by_name(tracking.EXPERIMENT) is not None


def test_explicit_uri_beats_the_environment(local_store: str, tmp_path: Path) -> None:
    other = f"sqlite:///{tmp_path / 'other.db'}"
    assert tracking.configure(other, env_file=None) == other


def test_real_environment_beats_dotenv(local_store: str, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MLFLOW_TRACKING_URI=http://should-not-win:5001\n")
    assert tracking.configure(env_file=env_file) == local_store


def test_runs_are_stamped_with_lineage(local_store: str, manifest: Path) -> None:
    tracking.configure(env_file=None)
    with (
        tracking.start_run("parent", manifest_path=manifest, tags={"stage": "test"}) as parent,
        tracking.start_run("child", nested=True, manifest_path=manifest) as child,
    ):
        mlflow.log_metric("pr_auc", 0.66)

    client = mlflow.MlflowClient()
    parent_tags = client.get_run(parent.info.run_id).data.tags
    child_run = client.get_run(child.info.run_id)
    assert parent_tags["data.train_sha256"] == "a" * 64
    assert parent_tags["stage"] == "test"
    assert child_run.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert child_run.data.tags["code.git_commit"] == parent_tags["code.git_commit"]
    assert child_run.data.metrics["pr_auc"] == 0.66


def test_unreachable_server_fails_fast_with_a_hint() -> None:
    # Port 9 (discard) is closed on any dev machine or CI runner: connection refused at once.
    with pytest.raises(tracking.TrackingServerUnavailable, match="docker compose up"):
        tracking.check_server("http://127.0.0.1:9", timeout=1)


def test_local_stores_need_no_server() -> None:
    tracking.check_server("sqlite:///anything.db")  # no error, no network


def test_content_fingerprint_ignores_how_the_file_was_written(tmp_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.to_parquet(tmp_path / "snappy.parquet", compression="snappy")
    df.to_parquet(tmp_path / "gzip.parquet", compression="gzip")
    snappy, gzipped = pd.read_parquet(tmp_path / "snappy.parquet"), pd.read_parquet(tmp_path / "gzip.parquet")
    assert lin.sha256_of(tmp_path / "snappy.parquet") != lin.sha256_of(tmp_path / "gzip.parquet")
    assert lin.content_sha256(snappy) == lin.content_sha256(gzipped)


def test_content_fingerprint_changes_when_the_data_changes() -> None:
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert lin.content_sha256(df) != lin.content_sha256(df.assign(a=[1, 2, 4]))
    assert lin.content_sha256(df) != lin.content_sha256(df.astype({"a": float}))  # types are part of the data
