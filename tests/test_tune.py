"""Tuning plumbing: search spaces produce valid models and a short study runs end to end."""

import optuna

from retention.config import TARGET
from retention.models.tune import FAMILIES, tune
from tests.test_train import small_dataset


def test_every_search_space_builds_a_model_that_trains() -> None:
    df = small_dataset()
    for family, (suggest, build, _) in FAMILIES.items():
        trial = optuna.trial.FixedTrial(
            {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_child_weight": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "reg_alpha": 0.01,
                "C": 1.0,
                "l1_ratio": 0.5,
            }
        )
        model = build(suggest(trial)).fit(df, df[TARGET])
        assert model.predict_proba(df).shape == (len(df), 2), family


def test_a_short_study_finds_a_best_score() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = tune(small_dataset(), "logistic regression", n_trials=3, folds=3)
    assert 0.0 <= study.best_value <= 1.0
    assert set(study.best_params) == {"C", "l1_ratio"}
