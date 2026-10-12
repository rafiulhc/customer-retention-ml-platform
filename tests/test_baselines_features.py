"""Baselines rank as the framing document says; preprocessing never leaks and never crashes."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from retention.config import EXCLUDED_FOR_FAIRNESS, ID_COLUMN, TARGET
from retention.models.baselines import ContractRule, ContractTenureRule, NobodyLeaves
from retention.models.features import model_pipeline, preprocessor
from tests.factories import clean_customers


def four_customers() -> pd.DataFrame:
    return clean_customers(
        {"Contract": "Month-to-month", "tenure": 2, "TotalCharges": "100"},  # 0: m2m, new
        {"Contract": "Month-to-month", "tenure": 40, "TotalCharges": "2000"},  # 1: m2m, long-standing
        {"Contract": "Two year", "tenure": 1, "TotalCharges": "50"},  # 2: committed, new
        {"Contract": "Two year", "tenure": 60, "TotalCharges": "3000", "Churn": "Yes"},  # 3: committed, old
    )


def churn_scores(model: object, df: pd.DataFrame) -> np.ndarray:
    fitted = clone(model).fit(df, df[TARGET])
    return fitted.predict_proba(df)[:, 1]


# ---------- Baselines ----------
def test_nobody_leaves_scores_everyone_zero() -> None:
    assert churn_scores(NobodyLeaves(), four_customers()).tolist() == [0, 0, 0, 0]


def test_contract_rule_flags_month_to_month_only() -> None:
    assert churn_scores(ContractRule(), four_customers()).tolist() == [1, 1, 0, 0]


def test_contract_tenure_rule_ranks_m2m_first_then_newest_first() -> None:
    scores = churn_scores(ContractTenureRule(), four_customers())
    ranking = np.argsort(-scores).tolist()
    assert ranking == [0, 1, 2, 3]  # m2m new > m2m old > committed new > committed old
    assert scores[1] > 0.5 > scores[2]  # the two contract groups never overlap


def test_baselines_return_valid_probabilities() -> None:
    df = four_customers()
    for model in (NobodyLeaves(), ContractRule(), ContractTenureRule()):
        proba = clone(model).fit(df, df[TARGET]).predict_proba(df)
        assert proba.shape == (4, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ---------- Preprocessing ----------
def test_preprocessor_never_outputs_id_target_or_excluded_columns() -> None:
    names = preprocessor(scale_numeric=True).fit(four_customers()).get_feature_names_out()
    for forbidden in [ID_COLUMN, TARGET, *EXCLUDED_FOR_FAIRNESS]:
        assert not any(name == forbidden or name.startswith(f"{forbidden}_") for name in names)


def test_scaler_uses_training_statistics_only() -> None:
    train, other = four_customers().iloc[:2], four_customers().iloc[2:]
    fitted = preprocessor(scale_numeric=True).fit(train)
    names = list(fitted.get_feature_names_out())
    tenure_train = fitted.transform(train)[:, names.index("tenure")]
    tenure_other = fitted.transform(other)[:, names.index("tenure")]
    assert np.isclose(tenure_train.mean(), 0.0)  # centred on the data it was fitted on...
    assert not np.isclose(tenure_other.mean(), 0.0)  # ...and NOT re-centred on new data: no leakage


def test_unknown_category_does_not_crash() -> None:
    train = four_customers()
    new = train.iloc[:1].copy()
    new["payment_method"] = new["payment_method"].cat.add_categories(["Crypto"])
    new.loc[new.index[0], "payment_method"] = "Crypto"
    pipe = model_pipeline(LogisticRegression(), scale_numeric=True).fit(train, train[TARGET])
    assert pipe.predict_proba(new).shape == (1, 2)
