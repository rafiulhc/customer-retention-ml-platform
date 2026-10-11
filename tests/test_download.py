"""Tests for the dataset download. No network: a local file:// URL stands in for the real server."""

import hashlib
from pathlib import Path

import pytest

from retention.data import download as dl

CONTENT = b"customerID,Churn\n0001,No\n0002,Yes\n"
SHA = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture
def source(tmp_path: Path) -> str:
    """A fake 'remote' dataset served from a local file."""
    src = tmp_path / "remote.csv"
    src.write_bytes(CONTENT)
    return src.as_uri()


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    assert dl.sha256_of(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_verify_rejects_wrong_content(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"tampered")
    with pytest.raises(dl.ChecksumError):
        dl.verify(f, SHA)


def test_download_writes_verified_file(tmp_path: Path, source: str) -> None:
    dest = tmp_path / "raw" / "data.csv"
    assert dl.download(source, dest, SHA) == dest
    assert dest.read_bytes() == CONTENT


def test_existing_verified_file_is_not_downloaded_again(tmp_path: Path) -> None:
    dest = tmp_path / "data.csv"
    dest.write_bytes(CONTENT)
    # an unreachable URL proves no download is attempted
    assert dl.download("http://127.0.0.1:9/never", dest, SHA) == dest


def test_corrupt_existing_file_is_replaced(tmp_path: Path, source: str) -> None:
    dest = tmp_path / "data.csv"
    dest.write_bytes(b"corrupted")
    dl.download(source, dest, SHA)
    assert dest.read_bytes() == CONTENT


def test_bad_download_leaves_no_file_behind(tmp_path: Path, source: str) -> None:
    dest = tmp_path / "data.csv"
    with pytest.raises(dl.ChecksumError):
        dl.download(source, dest, "0" * 64)
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []  # temporary file cleaned up too
