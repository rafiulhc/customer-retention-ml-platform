"""Module 5 building blocks, on synthetic data where the right answer is known."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from retention.config import FEATURES, TARGET
from retention.evaluation.bootstrap import paired_bootstrap
from retention.evaluation.calibration import calibration_table, expected_calibration_error
from retention.evaluation.explain import global_importance, shap_values, top_reasons
from retention.evaluation.oof import out_of_fold
from retention.evaluation.policy import Policy, choose_threshold, contact_mask, profit, profit_curve
from retention.evaluation.segments import largest_gap, segment_report
from retention.models.champion import model_input
from retention.models.train import final_model
from tests.test_train import small_dataset

RNG = np.random.default_rng(0)


def calibrated_sample(n: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    p = RNG.uniform(0, 1, n)
    return (RNG.uniform(0, 1, n) < p).astype(np.int64), p


# ---------- calibration ----------
def test_calibrated_scores_have_a_small_error_and_overconfident_ones_a_large_one() -> None:
    y, p = calibrated_sample()
    overconfident = np.clip(0.5 + 1.6 * (p - 0.5), 0, 1)  # pushed towards 0 and 1
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, overconfident) > 0.05


def test_calibration_table_has_equal_size_groups() -> None:
    y, p = calibrated_sample(1000)
    table = calibration_table(y, p, bins=10)
    assert len(table) == 10
    assert table["customers"].eq(100).all()


# ---------- decision policy ----------
def test_contact_mask_respects_capacity_and_threshold() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.1, 0.05, 0.6, 0.2, 0.4, 0.5])
    assert contact_mask(scores, Policy(threshold=0.0, capacity_fraction=0.3)).sum() == 3  # capacity binds
    mask = contact_mask(scores, Policy(threshold=0.85, capacity_fraction=0.3))
    assert mask.tolist() == [True] + [False] * 9  # threshold binds


def test_profit_counts_hits_and_wasted_calls() -> None:
    from retention.config import BUSINESS

    y = np.array([1, 0, 1, 0])
    contacted = np.array([True, True, True, False])
    assert profit(y, contacted) == pytest.approx(
        2 * BUSINESS.true_positive_value + BUSINESS.false_positive_value
    )


def test_profit_curve_ends_with_contacting_everyone() -> None:
    y, p = calibrated_sample(500)
    curve = profit_curve(y, p)
    assert len(curve) == 500
    assert curve["profit"].iloc[-1] == pytest.approx(profit(y, np.ones(500, dtype=bool)))


def test_with_calibrated_scores_the_chosen_threshold_is_near_break_even() -> None:
    """Without a capacity limit, profit is maximised around the break-even probability."""
    from retention.config import BUSINESS

    y, p = calibrated_sample(50_000)
    policy = choose_threshold(y, p, capacity_fraction=1.0)
    assert policy.threshold == pytest.approx(BUSINESS.break_even_probability, abs=0.03)


def test_with_a_capacity_limit_the_threshold_is_just_the_capacity_line() -> None:
    """Why the evaluation picks the threshold without the limit: with it, the answer isn't a decision rule."""
    y, p = calibrated_sample(50_000)
    capped = choose_threshold(y, p, capacity_fraction=0.10).threshold
    assert capped > 0.85  # ~ the score of the 10%-th customer, far above break-even (0.22)


# ---------- bootstrap ----------
def test_identical_models_are_not_clearly_better() -> None:
    y, p = calibrated_sample(500)
    c = paired_bootstrap(y, p, p, {"pr_auc": average_precision_score}, n_resamples=200)["pr_auc"]
    assert c.difference == 0 and not c.clearly_better


def test_an_informative_model_clearly_beats_noise() -> None:
    y, p = calibrated_sample(500)
    noise = RNG.uniform(0, 1, 500)
    c = paired_bootstrap(y, p, noise, {"pr_auc": average_precision_score}, n_resamples=200)["pr_auc"]
    assert c.clearly_better and c.ci_low <= c.difference <= c.ci_high


# ---------- segments ----------
def test_segment_report_shows_a_recall_gap() -> None:
    df = small_dataset(200)
    df["gender"] = np.where(np.arange(len(df)) % 2, "Male", "Female")
    y = df[TARGET].to_numpy()
    contacted = (y == 1) & (df["gender"] == "Female").to_numpy()  # female churners reached, male ones not
    report = segment_report(df, y, y.astype(float), contacted, columns=["gender"])
    assert largest_gap(report, "gender") == pytest.approx(1.0)
    assert set(report["group"]) == {"Female", "Male"}
    assert set(report.columns) >= {"recall", "precision", "contacted_share", "small_group"}


# ---------- explanations ----------
def test_shap_values_add_up_to_the_models_log_odds() -> None:
    df = small_dataset()
    X = model_input(df)
    model = final_model().fit(X, df[TARGET])
    values, base = shap_values(model, X, background=X)

    assert list(values.columns) == FEATURES  # one column per original feature, one-hots summed
    log_odds = model.decision_function(X)
    assert np.allclose(base + values.sum(axis=1).to_numpy(), log_odds)


def test_top_reasons_name_features_pushing_towards_leaving() -> None:
    df = small_dataset()
    X = model_input(df)
    model = final_model().fit(X, df[TARGET])
    values, _ = shap_values(model, X, background=X)
    reasons = top_reasons(values.head(5), X.head(5), k=3)
    assert len(reasons) == 5 and all(len(r) <= 3 for r in reasons)
    assert all(" = " in reason for r in reasons for reason in r)
    assert global_importance(values).index[0] in FEATURES


# ---------- out-of-fold ----------
def test_every_training_customer_gets_an_out_of_fold_score() -> None:
    df = small_dataset()
    scores = out_of_fold(df, final_model, folds=3)
    assert scores.shape == (len(df),) and not np.isnan(scores).any()


def test_out_of_fold_scores_differ_from_in_sample_scores() -> None:
    """Scored by models that never saw them, so they differ from a model fitted on everyone."""
    df = small_dataset()
    in_sample = final_model().fit(model_input(df), df[TARGET]).predict_proba(model_input(df))[:, 1]
    assert not np.allclose(out_of_fold(df, final_model, folds=3), in_sample)
