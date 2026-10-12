"""Tests for the train/test split and the data pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retention.data.clean import ID_COLUMN, TARGET
from retention.data.download import RAW_PATH
from retention.data.split import LeakageError, split


def customers(n: int = 1000, churn_rate: float = 0.27, seed: int = 0) -> pd.DataFrame:
    """A minimal 'cleaned' table: only the columns split() relies on."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            ID_COLUMN: [f"{i:04d}-TEST{chr(65 + i % 26)}" for i in range(n)],
            TARGET: (rng.random(n) < churn_rate).astype(int),
            "tenure": rng.integers(0, 72, n),
        }
    )


def test_sizes_are_80_20() -> None:
    parts = split(customers(1000))
    assert len(parts.train) == 800
    assert len(parts.test) == 200


def test_churn_rate_is_the_same_in_both_splits() -> None:
    df = customers(1000)
    parts = split(df)
    overall = df[TARGET].mean()
    assert parts.train[TARGET].mean() == pytest.approx(overall, abs=0.01)
    assert parts.test[TARGET].mean() == pytest.approx(overall, abs=0.01)


def test_no_customer_is_in_both_splits() -> None:
    parts = split(customers(1000))
    assert set(parts.train[ID_COLUMN]).isdisjoint(parts.test[ID_COLUMN])


def test_every_customer_is_kept() -> None:
    df = customers(1000)
    parts = split(df)
    assert set(parts.train[ID_COLUMN]) | set(parts.test[ID_COLUMN]) == set(df[ID_COLUMN])


def test_same_seed_gives_the_same_split() -> None:
    df = customers(1000)
    pd.testing.assert_frame_equal(split(df, seed=7).test, split(df, seed=7).test)


def test_different_seed_gives_a_different_split() -> None:
    df = customers(1000)
    assert set(split(df, seed=1).test[ID_COLUMN]) != set(split(df, seed=2).test[ID_COLUMN])


def test_duplicate_customer_ids_are_rejected() -> None:
    df = customers(10)
    df.loc[1, ID_COLUMN] = df.loc[0, ID_COLUMN]
    with pytest.raises(LeakageError):
        split(df)


@pytest.mark.skipif(
    not RAW_PATH.exists(), reason="dataset not downloaded (uv run python -m retention.data.download)"
)
def test_prepare_writes_parquet_and_manifest(tmp_path: Path) -> None:
    from retention.data.prepare import prepare

    manifest = prepare(out_dir=tmp_path)
    train = pd.read_parquet(tmp_path / "train.parquet")
    test = pd.read_parquet(tmp_path / "test.parquet")

    assert len(train) + len(test) == 7043
    assert isinstance(train["contract"].dtype, pd.CategoricalDtype)  # types survive the round trip
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert manifest["outputs"]["test"]["rows"] == len(test)
    assert manifest["raw"]["sha256"].startswith("16320c9c")
