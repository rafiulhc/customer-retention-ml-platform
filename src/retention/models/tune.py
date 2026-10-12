"""Hyperparameter tuning with Optuna, then an honest re-check on fresh folds.

Tuning picks whatever scores best on its folds, partly by luck, so its best score is optimistic.
After tuning, the tuned and default models are compared again on DIFFERENT folds (a new seed)
to see how much of the gain is real. The test set is still never touched.

Run:  uv run python -m retention.models.tune
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from functools import partial
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from retention.config import CV_FOLDS, SEED, TARGET
from retention.data.prepare import PROCESSED_DIR
from retention.models.experiment import log_cv_results, log_study
from retention.models.features import model_pipeline
from retention.models.train import REPORTS_DIR, candidates, cross_validate, paired_comparison
from retention.tracking import configure, start_run

logger = logging.getLogger(__name__)

Params = dict[str, Any]
FRESH_SEED = SEED + 1  # different folds for the honest re-check


# ---------- Model builders ----------
def xgboost(params: Params) -> Any:
    return model_pipeline(
        XGBClassifier(**params, eval_metric="logloss", random_state=SEED, n_jobs=-1), scale_numeric=False
    )


def logistic(params: Params) -> Any:
    # l1_ratio: 0 = L2 (ridge), 1 = L1 (lasso), in between = elastic net. saga supports all of them.
    return model_pipeline(
        LogisticRegression(**params, solver="saga", max_iter=5000, random_state=SEED), scale_numeric=True
    )


# ---------- Search spaces ----------
def suggest_xgboost(trial: optuna.Trial) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def suggest_logistic(trial: optuna.Trial) -> Params:
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


FAMILIES: dict[str, tuple[Callable[[optuna.Trial], Params], Callable[[Params], Any], int]] = {
    # name: (search space, builder, number of trials)
    "XGBoost": (suggest_xgboost, xgboost, 50),
    "logistic regression": (suggest_logistic, logistic, 30),
}


# ---------- Tuning ----------
def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    suggest: Callable[[optuna.Trial], Params],
    build: Callable[[Params], Any],
    folds: int = CV_FOLDS,
) -> float:
    """Mean cross-validated PR-AUC. Reported fold by fold so hopeless trials are stopped early."""
    params = suggest(trial)
    y = df[TARGET].to_numpy()
    scores: list[float] = []
    for fold, (tr, va) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(df, y)):
        model = clone(build(params)).fit(df.iloc[tr], y[tr])
        scores.append(float(average_precision_score(y[va], model.predict_proba(df.iloc[va])[:, 1])))
        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


def tune(df: pd.DataFrame, family: str, n_trials: int | None = None, folds: int = CV_FOLDS) -> optuna.Study:
    suggest, build, default_trials = FAMILIES[family]
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name=family,
    )
    study.optimize(lambda t: objective(t, df, suggest, build, folds), n_trials=n_trials or default_trials)
    return study


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    configure()
    with start_run("tuning", tags={"stage": "tuning"}) as run:
        mlflow.log_params({"train_rows": len(train), "fresh_seed": FRESH_SEED})
        report = _tune_and_recheck(train)
    logger.info("Logged to MLflow: run %s", run.info.run_id)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "tuning.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    logger.info("Report written to %s", REPORTS_DIR / "tuning.json")
    return 0


def _tune_and_recheck(train: pd.DataFrame) -> dict[str, Any]:
    """Tune every family, then re-check tuned vs default on fresh folds. Logs into the active run."""
    report: dict[str, Any] = {"seed": SEED, "fresh_seed": FRESH_SEED, "families": {}}
    tuned: dict[str, Callable[[], Any]] = {}
    for family, (_, build, n_trials) in FAMILIES.items():
        study = tune(train, family)
        log_study(study, n_trials=n_trials)
        pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        logger.info(
            "%s: best CV PR-AUC %.4f after %d trials (%d pruned)", family, study.best_value, n_trials, pruned
        )
        report["families"][family] = {
            "best_cv_pr_auc": study.best_value,
            "best_params": study.best_params,
            "trials": n_trials,
            "pruned": pruned,
        }
        tuned[f"{family} (tuned)"] = partial(build, study.best_params)

    # The honest re-check: defaults vs tuned, on folds the tuner never saw
    defaults = candidates()
    contenders = {
        "logistic regression (default)": defaults["B3 logistic regression"],
        "XGBoost (default)": defaults["XGBoost"],
        **tuned,
    }
    fresh = cross_validate(train, contenders, seed=FRESH_SEED)
    reference = "logistic regression (default)"
    summary = log_cv_results(fresh, contenders, reference=reference, cv_seed=FRESH_SEED)
    print(f"\nRe-check on fresh folds (seed {FRESH_SEED}), mean ± std over {CV_FOLDS} folds:")
    for model, row in summary.iterrows():
        line = f"  {model!s:<34} PR-AUC {row[('pr_auc', 'mean')]:.4f} ± {row[('pr_auc', 'std')]:.4f}"
        if model != reference:
            c = paired_comparison(fresh, str(model), reference=reference)["pr_auc"]
            line += f"   vs default logistic: {c['mean_difference']:+.4f} ± {c['std_difference']:.4f}"
            line += f", better on {c['folds_won']}/{c['folds']} folds"
        print(line)

    report["fresh_folds"] = fresh.round(4).to_dict(orient="records")
    mlflow.log_dict(report, "tuning.json")
    return report


if __name__ == "__main__":
    sys.exit(main())
