"""The CI model check: unchanged code passes, a regression fails, an improvement asks for a new baseline."""

from typing import Any

from retention.models.regression_check import PR_AUC_TOLERANCE, compare, measure, report
from tests.factories import learnable_dataset

BASELINE: dict[str, Any] = {
    "folds": 5,
    "data_fingerprint": "abc",
    "pr_auc_mean": 0.66,
    "profit_at_k_mean": 16_000.0,
    "folds_beating_b2_pr_auc": 5,
    "folds_beating_b2_profit": 5,
}


def failed(current: dict[str, Any]) -> set[str]:
    return {c.name for c in compare(current, BASELINE) if not c.passed}


def test_the_same_numbers_pass() -> None:
    assert failed(dict(BASELINE)) == set()


def test_a_drop_beyond_the_tolerance_fails() -> None:
    assert failed({**BASELINE, "pr_auc_mean": 0.66 - 2 * PR_AUC_TOLERANCE}) == {
        "PR-AUC not below the baseline"
    }
    assert "profit@10% not below the baseline" in failed({**BASELINE, "profit_at_k_mean": 15_000.0})


def test_losing_to_the_rule_on_any_fold_fails() -> None:
    assert failed({**BASELINE, "folds_beating_b2_profit": 4}) == {"beats the B2 rule on every fold"}


def test_different_data_is_flagged_as_not_comparable() -> None:
    assert failed({**BASELINE, "data_fingerprint": "xyz"}) == {"same training data as the baseline"}


def test_an_improvement_passes_and_asks_for_a_new_baseline() -> None:
    better = {**BASELINE, "pr_auc_mean": 0.70}
    assert failed(better) == set()
    assert "update the baseline" in report(compare(better, BASELINE), better, BASELINE)


def test_measure_is_deterministic() -> None:
    df = learnable_dataset(300, seed=1)
    assert measure(df) == measure(df)
