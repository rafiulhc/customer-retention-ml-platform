"""Download the public IBM Telco Customer Churn dataset into data/raw/ and verify its checksum.

Run:  uv run python -m retention.data.download
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/retention/data/download.py -> repository root
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


class ChecksumError(Exception):
    """The file's contents are not the expected dataset."""


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so large files don't need to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_sha256: str) -> None:
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise ChecksumError(f"{path.name}: expected sha256 {expected_sha256}, got {actual}")


def download(
    url: str = DATASET_URL,
    dest: Path = RAW_PATH,
    expected_sha256: str = EXPECTED_SHA256,
    timeout: float = 60.0,
) -> Path:
    """Download `url` to `dest` unless a verified copy already exists.

    The file is written to a temporary name and only moved into place after its checksum passes,
    so a failed or corrupted download never leaves a bad file behind.
    """
    if dest.exists():
        try:
            verify(dest, expected_sha256)
        except ChecksumError:
            logger.warning("Existing %s failed its checksum; downloading again", dest.name)
        else:
            logger.info("%s already present and verified; skipping download", dest.name)
            return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout) as response:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        verify(tmp, expected_sha256)
        tmp.replace(dest)  # atomic on the same filesystem
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("Downloaded and verified %s", dest)
    return dest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        download()
    except (ChecksumError, OSError) as e:
        logger.error("Download failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
