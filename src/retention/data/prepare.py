"""The data pipeline: download -> validate & clean -> split -> Parquet files + manifest.

Run:  uv run python -m retention.data.prepare
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from retention.data.clean import TARGET, clean
from retention.data.download import PROJECT_ROOT, RAW_PATH, download, sha256_of
from retention.data.schema import CLEAN_SCHEMA
from retention.data.split import SEED, TEST_SIZE, split
from retention.lineage import git_commit

logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _summary(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "rows": len(df),
        "churn_rate": round(float(df[TARGET].mean()), 4),
        "sha256": sha256_of(path),
    }


def prepare(raw_path: Path = RAW_PATH, out_dir: Path = PROCESSED_DIR, seed: int = SEED) -> dict[str, Any]:
    """Build train/test Parquet files from the raw CSV and return the manifest."""
    cleaned = clean(pd.read_csv(raw_path))
    parts = split(cleaned, test_size=TEST_SIZE, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, df in (("train", parts.train), ("test", parts.test)):
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        CLEAN_SCHEMA.validate(pd.read_parquet(path), lazy=True)  # the saved file round-trips with its types
        files[name] = _summary(df, path)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": git_commit(),
        "raw": {"file": raw_path.name, "sha256": sha256_of(raw_path), "rows": len(cleaned)},
        "split": {"test_size": TEST_SIZE, "seed": seed, "stratified_on": TARGET},
        "outputs": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        download()
        manifest = prepare()
    except Exception:
        logger.exception("Data preparation failed")
        return 1
    for name, info in manifest["outputs"].items():
        logger.info("%-5s %5d rows  churn %.4f  -> %s", name, info["rows"], info["churn_rate"], info["file"])
    logger.info("Manifest written to %s", PROCESSED_DIR / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
