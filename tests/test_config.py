"""The business numbers from docs/problem_framing.md, and the feature list rules."""

import pytest

from retention.config import (
    BUSINESS,
    CATEGORICAL_FEATURES,
    EXCLUDED_FOR_FAIRNESS,
    FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET,
)
from retention.data.schema import CATEGORY_COLUMNS, CLEAN_SCHEMA


def test_business_values_match_the_framing_document() -> None:
    assert BUSINESS.value_of_saving == pytest.approx(893.28)
    assert BUSINESS.true_positive_value == pytest.approx(207.98, abs=0.01)
    assert BUSINESS.false_positive_value == -60.0
    assert BUSINESS.break_even_probability == pytest.approx(0.224, abs=0.001)


def test_features_never_include_the_id_target_or_excluded_columns() -> None:
    for column in [ID_COLUMN, TARGET, *EXCLUDED_FOR_FAIRNESS]:
        assert column not in FEATURES


def test_every_feature_exists_in_the_clean_data() -> None:
    assert set(FEATURES) <= set(CLEAN_SCHEMA.columns)
    assert set(CATEGORICAL_FEATURES) <= set(CATEGORY_COLUMNS)
    assert len(FEATURES) == len(set(FEATURES)) == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
