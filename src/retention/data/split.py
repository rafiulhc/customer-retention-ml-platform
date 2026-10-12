"""Stratified, seeded, leakage-checked train/test split.

The test set is the final, honest estimate: it is used once, at the end of evaluation (module 5).
Model selection uses cross-validation inside the training set instead.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from retention.data.clean import ID_COLUMN, TARGET

TEST_SIZE = 0.2
SEED = 42


class LeakageError(Exception):
    """The same customer would end up in both train and test."""


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED) -> Split:
    """Split cleaned data, keeping the churn rate equal in both parts (stratified)."""
    if df[ID_COLUMN].duplicated().any():
        raise LeakageError("duplicate customer IDs: the same customer could land in both splits")

    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])

    overlap = set(train[ID_COLUMN]) & set(test[ID_COLUMN])
    if overlap:  # defensive: guaranteed by the duplicate check, but cheap and it documents the rule
        raise LeakageError(f"{len(overlap)} customers are in both train and test")

    return Split(train=train.reset_index(drop=True), test=test.reset_index(drop=True))
