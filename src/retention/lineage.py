"""Lineage: which code and which data produced a result.

Every MLflow run is stamped with these values so any number in the UI can be traced back to an
exact commit and exact data files (by checksum), and reproduced.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from retention.data.download import PROJECT_ROOT, sha256_of

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.json"


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


# Container images have no git: the build records the commit in these variables instead
# (`--build-arg GIT_COMMIT=...`, see infra/pipeline/Dockerfile).
COMMIT_ENV, DIRTY_ENV = "RETENTION_GIT_COMMIT", "RETENTION_GIT_DIRTY"


def git_commit() -> str | None:
    """The current commit (short hash): from git, or from the image build; None if neither knows."""
    return _git("rev-parse", "--short", "HEAD") or os.environ.get(COMMIT_ENV) or None


def git_dirty() -> bool | None:
    """True if tracked files have uncommitted changes: the commit alone can't reproduce the run."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is not None:
        return bool(status)
    recorded = os.environ.get(DIRTY_ENV)
    return None if recorded is None else recorded == "true"


def content_sha256(df: pd.DataFrame) -> str:
    """A fingerprint of the data itself: column names, types and values, in order.

    Unlike the file's checksum, it doesn't change when the same data is written by a different library
    version (Parquet files record the writer's version), so it answers "is this the same data?".
    """
    digest = hashlib.sha256(json.dumps([[str(c), str(t)] for c, t in df.dtypes.items()]).encode())
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def same_training_data(version_tags: Mapping[str, str], train_path: Path) -> bool:
    """Was a model version trained on the data in `train_path`? Compares content when the version has a
    content fingerprint, otherwise the file checksum (versions registered before fingerprints existed)."""
    if "data.train_content_sha256" in version_tags:
        return version_tags["data.train_content_sha256"] == content_sha256(pd.read_parquet(train_path))
    return version_tags.get("data.train_sha256") == sha256_of(train_path)


def data_lineage(manifest_path: Path = MANIFEST_PATH) -> dict[str, str]:
    """Checksums of the raw and processed data, read from the manifest written by `prepare`."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run `uv run python -m retention.data.prepare` first."
        )
    manifest = json.loads(manifest_path.read_text())
    tags = {
        "data.raw_sha256": manifest["raw"]["sha256"],
        "data.prepared_at": manifest["created_at"],
        "data.prepared_by_commit": str(manifest["code_version"]),
    }
    for name, info in manifest["outputs"].items():
        tags[f"data.{name}_sha256"] = info["sha256"]
        tags[f"data.{name}_rows"] = str(info["rows"])
        if "content_sha256" in info:  # manifests written before fingerprints existed don't have it
            tags[f"data.{name}_content_sha256"] = info["content_sha256"]
    return tags


def lineage(manifest_path: Path = MANIFEST_PATH) -> dict[str, str]:
    """Code + data lineage as string tags (MLflow tags are strings)."""
    dirty = git_dirty()
    return {
        "code.git_commit": git_commit() or "unknown",
        "code.git_dirty": "unknown" if dirty is None else str(dirty).lower(),
        **data_lineage(manifest_path),
    }
