"""Ranking metrics under a capacity limit, checked against examples you can work out by hand."""

import numpy as np
import pytest

from retention.config import BUSINESS
from retention.models.metrics import (
    evaluate,
    expected_profit_at_k,
    precision_at_k,
    recall_at_k,
    top_k_mask,
)

#            A    B    C    D    E    F    G    H    I    J
Y = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.05, 0.05, 0.02, 0.01, 0.0])


def test_top_k_selects_the_highest_scores() -> None:
    assert top_k_mask(S, 0.2).tolist() == [True, True] + [False] * 8


def test_top_k_always_selects_at_least_one() -> None:
    assert top_k_mask(S, 0.001).sum() == 1


def test_ties_are_broken_by_original_order() -> None:
    # all scores equal: the first customers win, so results are deterministic
    assert top_k_mask(np.zeros(5), 0.4).tolist() == [True, True, False, False, False]


def test_precision_at_k() -> None:
    assert precision_at_k(Y, S, 0.2) == 0.5  # A churns, B doesn't


def test_recall_at_k() -> None:
    assert recall_at_k(Y, S, 0.2) == pytest.approx(1 / 3)  # 1 of the 3 churners reached


def test_expected_profit_at_k() -> None:
    expected = BUSINESS.true_positive_value + BUSINESS.false_positive_value  # one useful, one wasted call
    assert expected_profit_at_k(Y, S, 0.2) == pytest.approx(expected)


def test_perfect_ranking() -> None:
    perfect = Y.astype(float)  # churners get 1.0, everyone else 0.0
    assert precision_at_k(Y, perfect, 0.3) == 1.0
    assert recall_at_k(Y, perfect, 0.3) == 1.0


def test_evaluate_returns_every_metric() -> None:
    result = evaluate(Y, S, 0.2)
    assert set(result) == {"pr_auc", "roc_auc", "brier", "precision_at_k", "recall_at_k", "profit_at_k"}
    assert result["precision_at_k"] == 0.5


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        precision_at_k(Y, S[:5], 0.2)


def test_missing_scores_are_rejected() -> None:
    with pytest.raises(ValueError):
        precision_at_k(Y, np.array([np.nan] * 10), 0.2)
