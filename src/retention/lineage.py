"""Lineage: which code and which data produced a result.

Every MLflow run is stamped with these values so any number in the UI can be traced back to an
exact commit and exact data files (by checksum), and reproduced.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from retention.data.download import PROJECT_ROOT

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.json"


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_commit() -> str | None:
    """The current commit (short hash), or None outside a git checkout."""
    return _git("rev-parse", "--short", "HEAD")


def git_dirty() -> bool | None:
    """True if tracked files have uncommitted changes: the commit alone can't reproduce the run."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


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
    return tags


def lineage(manifest_path: Path = MANIFEST_PATH) -> dict[str, str]:
    """Code + data lineage as string tags (MLflow tags are strings)."""
    dirty = git_dirty()
    return {
        "code.git_commit": git_commit() or "unknown",
        "code.git_dirty": "unknown" if dirty is None else str(dirty).lower(),
        **data_lineage(manifest_path),
    }
