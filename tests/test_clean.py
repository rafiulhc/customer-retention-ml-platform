"""Tests for clean(): small hand-made rows, so each test shows exactly one behaviour."""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from retention.data.clean import ID_COLUMN, TARGET, clean
from retention.data.download import RAW_PATH
from retention.data.schema import CATEGORY_COLUMNS
from tests.factories import frame, raw_row


def test_blank_total_charges_for_new_customer_becomes_zero() -> None:
    df = clean(frame(raw_row(tenure=0, TotalCharges=" ")))
    assert df["total_charges"].iloc[0] == 0.0


def test_blank_total_charges_for_existing_customer_is_rejected() -> None:
    with pytest.raises(SchemaErrors):
        clean(frame(raw_row(tenure=8, TotalCharges=" ")))


def test_total_charges_becomes_a_number() -> None:
    df = clean(frame(raw_row(TotalCharges="1234.5")))
    assert df["total_charges"].iloc[0] == 1234.5
    assert pd.api.types.is_float_dtype(df["total_charges"])


def test_whole_number_total_charges_stay_float() -> None:
    # regression test: to_numeric(["100"]) returns int64, which the clean schema used to reject
    df = clean(frame(raw_row(TotalCharges="100")))
    assert df["total_charges"].iloc[0] == 100.0
    assert pd.api.types.is_float_dtype(df["total_charges"])


def test_churn_becomes_0_and_1() -> None:
    df = clean(
        frame(raw_row(customerID="0001-AAAAA", Churn="Yes"), raw_row(customerID="0002-BBBBB", Churn="No"))
    )
    assert df[TARGET].tolist() == [1, 0]
    assert pd.api.types.is_integer_dtype(df[TARGET])


def test_senior_citizen_becomes_yes_no() -> None:
    df = clean(frame(raw_row(customerID="0001-AAAAA", SeniorCitizen=1), raw_row(customerID="0002-BBBBB")))
    assert df["senior_citizen"].tolist() == ["Yes", "No"]


def test_columns_are_snake_case_and_categories_are_typed() -> None:
    df = clean(frame(raw_row()))
    assert all(name == name.lower() for name in df.columns)
    assert "customerID" not in df.columns and ID_COLUMN in df.columns
    assert all(isinstance(df[c].dtype, pd.CategoricalDtype) for c in CATEGORY_COLUMNS)


def test_clean_does_not_change_its_input() -> None:
    raw = frame(raw_row(tenure=0, TotalCharges=" ", SeniorCitizen=1, Churn="Yes"))
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_unknown_category_is_rejected() -> None:
    with pytest.raises(SchemaErrors):
        clean(frame(raw_row(Contract="Weekly")))


@pytest.mark.skipif(
    not RAW_PATH.exists(), reason="dataset not downloaded (uv run python -m retention.data.download)"
)
def test_real_dataset_cleans_end_to_end() -> None:
    df = clean(pd.read_csv(RAW_PATH))
    assert df.shape == (7043, 21)
    assert df[TARGET].mean() == pytest.approx(0.2654, abs=1e-4)
    assert df.isna().sum().sum() == 0
