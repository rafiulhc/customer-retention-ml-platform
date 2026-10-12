"""Model comparisons logged to MLflow: a parent run with the verdict, one child run per model."""

import mlflow
import optuna
import pytest
from mlflow.entities import Run

from retention.models.experiment import log_cv_results, log_study, model_params
from retention.models.train import candidates, cross_validate
from retention.models.tune import tune
from retention.tracking import start_run
from tests.test_train import RULES, small_dataset

MODELS = {**RULES, "B3 logistic regression": candidates()["B3 logistic regression"]}


def children(parent_id: str) -> dict[str, Run]:
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.parentRunId = '{parent_id}'", output_format="list")
    return {r.info.run_name: r for r in runs}


def test_model_params_come_from_the_final_pipeline_step() -> None:
    params = model_params(candidates()["B3 logistic regression"]())
    assert params["model_class"] == "LogisticRegression"
    assert params["max_iter"] == 2000
    assert all(v is None or isinstance(v, str | int | float | bool) for v in params.values())


def test_logging_needs_a_parent_run(tracked: str) -> None:
    with pytest.raises(RuntimeError, match="parent run"):
        log_cv_results(cross_validate(small_dataset(), RULES, folds=3), RULES, reference="B2", cv_seed=42)


def test_one_child_run_per_model_with_means_and_per_fold_scores(tracked: str) -> None:
    results = cross_validate(small_dataset(), MODELS, folds=3)
    with start_run("selection") as parent:
        summary = log_cv_results(results, MODELS, reference="B2", cv_seed=42, final="B3 logistic regression")

    runs = children(parent.info.run_id)
    assert set(runs) == set(MODELS)
    final = runs["B3 logistic regression"]
    assert final.data.tags["is_final"] == "true"
    assert runs["B2"].data.tags["is_reference"] == "true"
    assert final.data.params["model_class"] == "LogisticRegression"
    assert final.data.tags["data.train_sha256"] == "a" * 64  # lineage reaches the children too
    assert final.data.metrics["pr_auc"] == pytest.approx(
        summary.loc["B3 logistic regression", ("pr_auc", "mean")]
    )

    history = mlflow.MlflowClient().get_metric_history(final.info.run_id, "fold_pr_auc")
    assert [m.step for m in history] == [0, 1, 2]

    ranks = sorted(int(r.data.tags["rank"]) for r in runs.values())
    assert ranks == list(range(1, len(MODELS) + 1))


def test_parent_run_holds_the_verdict_and_artifacts(tracked: str) -> None:
    results = cross_validate(small_dataset(), MODELS, folds=3)
    with start_run("selection") as parent:
        summary = log_cv_results(results, MODELS, reference="B0", cv_seed=42)

    run = mlflow.get_run(parent.info.run_id)
    assert run.data.tags["best_model"] == summary.index[0]
    assert run.data.params["primary_metric"] == "pr_auc"
    assert "best_vs_reference_pr_auc_diff" in run.data.metrics
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(parent.info.run_id)}
    assert {"cv_summary.txt", "cv_per_fold.csv"} <= artifacts


def test_an_optuna_study_becomes_a_child_run(tracked: str) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = tune(small_dataset(), "logistic regression", n_trials=3, folds=3)
    with start_run("tuning") as parent:
        log_study(study, n_trials=3)

    run = children(parent.info.run_id)["search: logistic regression"]
    assert run.data.metrics["best_cv_pr_auc"] == pytest.approx(study.best_value)
    assert set(study.best_params) <= set(run.data.params)
    assert [a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)] == ["trials.csv"]
