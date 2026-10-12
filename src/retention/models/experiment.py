"""Log model comparisons to MLflow: one parent run per comparison, one nested child run per model.

The parent holds the verdict (best model, how it does against the reference) and the full results
as artifacts. Each child holds one model's hyperparameters, its cross-validated means as metrics,
and its per-fold scores as stepped metrics, so models can be compared side by side in the UI.

Metric names in child runs:
    pr_auc, pr_auc_std, ...   mean and std over the folds (what the comparison table shows)
    fold_pr_auc, ...          one value per fold, logged with step = fold (the charts)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import mlflow
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline

from retention.models.train import PRIMARY_METRIC, format_table, paired_comparison, summarise
from retention.tracking import start_run

Param = str | int | float | bool | None


def model_params(model: Any) -> dict[str, Param]:
    """The estimator's own hyperparameters (the last pipeline step), plain values only."""
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
    params: dict[str, Param] = {"model_class": type(estimator).__name__}
    for key, value in estimator.get_params(deep=False).items():
        if value is None or isinstance(value, str | int | float | bool):
            params[key] = value
    return params


def _metric_names(results: pd.DataFrame) -> list[str]:
    return [c for c in results.columns if c not in ("model", "fold")]


def log_cv_results(
    results: pd.DataFrame,
    models: Mapping[str, Callable[[], Any]],
    *,
    reference: str,
    cv_seed: int,
    final: str | None = None,
) -> pd.DataFrame:
    """Log one child run per model under the active parent run, then the verdict on the parent.

    `results` is the output of `cross_validate` (one row per model and fold). Returns the summary.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("log_cv_results needs an active parent run (open one with start_run).")
    metrics = _metric_names(results)
    summary = summarise(results)
    folds = int(results["fold"].nunique())

    for rank, (name, row) in enumerate(summary.iterrows(), start=1):
        name = str(name)
        tags = {
            "role": "candidate",
            "rank": str(rank),
            "is_reference": str(name == reference).lower(),
            "is_final": str(name == final).lower(),
        }
        with start_run(name, nested=True, tags=tags):
            mlflow.log_params({"cv_seed": cv_seed, "cv_folds": folds, **model_params(models[name]())})
            mlflow.log_metrics(
                {
                    **{m: float(row[(m, "mean")]) for m in metrics},
                    **{f"{m}_std": float(row[(m, "std")]) for m in metrics},
                }
            )
            per_fold = results[results["model"] == name].sort_values("fold")
            for record in per_fold.to_dict(orient="records"):
                mlflow.log_metrics({f"fold_{m}": float(record[m]) for m in metrics}, step=int(record["fold"]))

    best = str(summary.index[0])
    mlflow.set_tags({"best_model": best, "reference": reference, **({"final_model": final} if final else {})})
    mlflow.log_params({"cv_seed": cv_seed, "cv_folds": folds, "primary_metric": PRIMARY_METRIC})
    mlflow.log_metric(f"best_{PRIMARY_METRIC}", float(summary.iloc[0][(PRIMARY_METRIC, "mean")]))
    if best != reference:
        for metric, c in paired_comparison(results, best, reference=reference).items():
            mlflow.log_metrics(
                {
                    f"best_vs_reference_{metric}_diff": c["mean_difference"],
                    f"best_vs_reference_{metric}_folds_won": c["folds_won"],
                }
            )
    mlflow.log_text(format_table(summary), "cv_summary.txt")
    mlflow.log_text(results.to_csv(index=False), "cv_per_fold.csv")
    return summary


def log_study(study: optuna.Study, *, n_trials: int) -> None:
    """Log one Optuna search as a child run: the best parameters, its score, and every trial."""
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    with start_run(f"search: {study.study_name}", nested=True, tags={"role": "search"}):
        mlflow.log_params({"n_trials": n_trials, **study.best_params})
        mlflow.log_metrics(
            {
                "best_cv_pr_auc": study.best_value,
                "pruned_trials": pruned,
                "best_trial": study.best_trial.number,
            }
        )
        mlflow.log_text(study.trials_dataframe().to_csv(index=False), "trials.csv")
