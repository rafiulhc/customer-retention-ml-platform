"""Cross-validation plumbing: every model on the same folds, results summarised and compared correctly."""

import numpy as np
import pandas as pd

from retention.config import TARGET
from retention.models.baselines import ContractRule, ContractTenureRule, NobodyLeaves
from retention.models.train import candidates, cross_validate, paired_comparison, summarise
from tests.factories import clean_customers


def small_dataset(n: int = 60) -> pd.DataFrame:
    """Month-to-month customers churn more, so the rules have something to find."""
    rng = np.random.default_rng(0)
    specs = []
    for _ in range(n):
        m2m = rng.random() < 0.5
        tenure = int(rng.integers(0, 72))
        churn = rng.random() < (0.6 if m2m else 0.1)
        specs.append(
            {
                "Contract": "Month-to-month" if m2m else "Two year",
                "tenure": tenure,
                "TotalCharges": str(50.0 * tenure) if tenure else " ",
                "Churn": "Yes" if churn else "No",
            }
        )
    return clean_customers(*specs)


RULES = {"B0": NobodyLeaves, "B1": ContractRule, "B2": ContractTenureRule}


def test_every_model_is_scored_on_every_fold() -> None:
    results = cross_validate(small_dataset(), RULES, folds=3)
    assert len(results) == 3 * 3
    assert results.groupby("model")["fold"].nunique().eq(3).all()


def test_summary_is_sorted_by_the_primary_metric() -> None:
    summary = summarise(cross_validate(small_dataset(), RULES, folds=3))
    means = summary[("pr_auc", "mean")].to_numpy()
    assert (means[:-1] >= means[1:]).all()
    assert summary.index[-1] == "B0"  # scoring nobody can't beat ranking anyone


def test_paired_comparison_counts_folds_won() -> None:
    results = cross_validate(small_dataset(), RULES, folds=3)
    c = paired_comparison(results, "B1", reference="B0")
    assert c["pr_auc"]["folds"] == 3
    assert c["pr_auc"]["folds_won"] == 3  # the contract rule beats "nobody leaves" on every fold


def test_all_candidates_train_and_predict() -> None:
    df = small_dataset()
    for name, make in candidates().items():
        proba = make().fit(df, df[TARGET]).predict_proba(df)
        assert proba.shape == (len(df), 2), name
