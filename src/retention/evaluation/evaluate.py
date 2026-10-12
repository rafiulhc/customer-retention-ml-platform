"""Module 5: the one-time evaluation of the champion on the held-out test set.

Order matters:
  1. Decisions first, on out-of-fold predictions from the TRAINING set: calibration check and the
     contact threshold. Nothing here sees the test set.
  2. Then the test set, once: metrics with 95% intervals against the B2 rule, the success criteria
     from docs/problem_framing.md, the policy's profit, results per segment, and SHAP explanations.
  3. Everything is logged to MLflow, and the verdict and threshold are written onto the model version,
     so serving uses the threshold that was evaluated.

A model version is evaluated on the test set only once; `--force` re-runs it (same model and data
give the same numbers, but repeated looks at the test set are how overfitting to it starts).

Run:  uv run python -m retention.evaluation.evaluate [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from retention.config import BUSINESS, CONTACT_FRACTION, TARGET
from retention.data.download import sha256_of
from retention.data.prepare import PROCESSED_DIR
from retention.evaluation import plots
from retention.evaluation.bootstrap import Comparison, paired_bootstrap
from retention.evaluation.calibration import (
    calibration_in_the_large,
    calibration_table,
    expected_calibration_error,
)
from retention.evaluation.explain import global_importance, shap_values, top_reasons
from retention.evaluation.oof import out_of_fold
from retention.evaluation.policy import Policy, choose_threshold, contact_mask, profit, profit_curve
from retention.evaluation.segments import largest_gap, segment_report
from retention.models.baselines import ContractTenureRule
from retention.models.champion import CHAMPION, MODEL_NAME, model_input
from retention.models.metrics import evaluate, expected_profit_at_k
from retention.models.train import final_model
from retention.tracking import EXPERIMENT, configure, start_run

logger = logging.getLogger(__name__)

STAGE = "test-evaluation"
CALIBRATION_TOLERANCE = 0.03  # ECE above this would mean recalibrating before using p > break-even
FAIRNESS_TOLERANCE = 0.10  # recall gap between groups worth investigating


class AlreadyEvaluated(RuntimeError):
    """This model version has already been scored on the test set."""


def decide(train: pd.DataFrame) -> dict[str, Any]:
    """Every decision, from out-of-fold training predictions only. The test set is not touched."""
    y = train[TARGET].to_numpy()
    oof = out_of_fold(train, final_model)
    policy = choose_threshold(y, oof)
    curve = profit_curve(y, oof)
    best = int(np.argmax(curve["profit"].to_numpy()))
    return {
        "oof_scores": oof,
        "calibration_table": calibration_table(y, oof),
        "ece": expected_calibration_error(y, oof),
        "calibration_in_the_large": calibration_in_the_large(y, oof),
        "policy": policy,
        "profit_curve": curve,
        # with unlimited capacity: how many to contact, and what that would earn (per training-set size)
        "uncapped_best_share": float(curve["share"].iloc[best]),
        "uncapped_best_profit": float(curve["profit"].iloc[best]),
        "capped_profit": profit(y, contact_mask(oof, policy)),
    }


def success_criteria(comparisons: dict[str, Comparison]) -> dict[str, bool]:
    """docs/problem_framing.md section 5: beat B2 on PR-AUC and on profit for the top 10%.

    "Beat" is read strictly: the whole 95% interval of the difference must be above zero.
    The champion is B3 itself, so the "beat B3 clearly, or ship B3" clause is satisfied by construction.
    """
    return {
        "beats_b2_on_pr_auc": comparisons["pr_auc"].clearly_better,
        "beats_b2_on_profit_at_k": comparisons["profit_at_k"].clearly_better,
    }


def _check_once(client: MlflowClient, version: ModelVersion, force: bool) -> None:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None or force:
        return
    previous = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.stage = '{STAGE}' and tags.model_version = '{version.version}'",
    )
    if previous:
        raise AlreadyEvaluated(
            f"version {version.version} was already evaluated on the test set "
            f"(run {previous[0].info.run_id}). Use --force to re-run it."
        )


def run(force: bool = False) -> dict[str, Any]:
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    _check_once(client, version, force)

    train_path, test_path = PROCESSED_DIR / "train.parquet", PROCESSED_DIR / "test.parquet"
    if sha256_of(train_path) != version.tags.get("data.train_sha256"):
        raise RuntimeError(
            "The champion was trained on different data than train.parquet; re-register it first."
        )
    train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)

    # 1. Decisions (training data only)
    decisions = decide(train)
    policy: Policy = decisions["policy"]

    # 2. The test set, once
    model: Pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version.version}")
    X_test, y_test = model_input(test), test[TARGET].to_numpy()
    scores = model.predict_proba(X_test)[:, 1]
    rule = ContractTenureRule().fit(model_input(train), train[TARGET])
    rule_scores = rule.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, scores)
    rule_metrics = evaluate(y_test, rule_scores)
    comparisons = paired_bootstrap(
        y_test,
        scores,
        rule_scores,
        {"pr_auc": average_precision_score, "profit_at_k": expected_profit_at_k},
    )
    criteria = success_criteria(comparisons)

    contacted = contact_mask(scores, policy)
    policy_result = {
        "contacted": int(contacted.sum()),
        "contacted_share": float(contacted.mean()),
        "churners_reached": int(y_test[contacted].sum()),
        "precision": float(y_test[contacted].mean()) if contacted.any() else 0.0,
        "recall": float(y_test[contacted].sum() / y_test.sum()),
        "profit": profit(y_test, contacted),
        "profit_per_contact": profit(y_test, contacted) / max(1, int(contacted.sum())),
    }
    test_curve = profit_curve(y_test, scores)

    segments = segment_report(test, y_test, scores, contacted)
    fairness = {f"recall_gap_{s}": largest_gap(segments, s) for s in ("gender", "senior_citizen")}

    shap_test, base_value = shap_values(model, X_test, background=model_input(train))
    importance = global_importance(shap_test)
    flagged = np.flatnonzero(contacted)[np.argsort(-scores[contacted])][:5]
    examples = [
        {"rank": rank + 1, "p_leaves": round(float(scores[i]), 3), "reasons": reasons}
        for rank, (i, reasons) in enumerate(
            zip(flagged, top_reasons(shap_test.iloc[flagged], X_test.iloc[flagged]), strict=True)
        )
    ]

    report: dict[str, Any] = {
        "model": {
            "name": MODEL_NAME,
            "version": str(version.version),
            "family": version.tags.get("model_family"),
        },
        "decisions_on_training_oof": {
            "ece": decisions["ece"],
            "calibration_in_the_large": decisions["calibration_in_the_large"],
            "calibrated_enough": decisions["ece"] <= CALIBRATION_TOLERANCE,
            "policy": policy.as_dict(),
            "break_even_probability": BUSINESS.break_even_probability,
            "uncapped_best_share": decisions["uncapped_best_share"],
            "profit_within_capacity": decisions["capped_profit"],
            "profit_uncapped": decisions["uncapped_best_profit"],
        },
        "test": {
            "customers": len(test),
            "churn_rate": float(y_test.mean()),
            "champion": metrics,
            "b2_rule": rule_metrics,
            "vs_b2": {name: asdict(c) for name, c in comparisons.items()},
            "success_criteria": criteria,
            "passed": all(criteria.values()),
            "calibration": {
                "ece": expected_calibration_error(y_test, scores),
                "calibration_in_the_large": calibration_in_the_large(y_test, scores),
            },
            "policy": policy_result,
            "uncapped_best_share": float(
                test_curve["share"].iloc[int(np.argmax(test_curve["profit"].to_numpy()))]
            ),
            "fairness": fairness,
        },
        "explanations": {
            "base_log_odds": base_value,
            "global_importance": importance.round(4).to_dict(),
            "top_flagged_examples": examples,
        },
    }

    _log(version, report, decisions, test_curve, segments, importance, y_test, scores, rule_scores)
    return report


def _log(
    version: ModelVersion,
    report: dict[str, Any],
    decisions: dict[str, Any],
    test_curve: pd.DataFrame,
    segments: pd.DataFrame,
    importance: pd.Series,
    y_test: np.ndarray,
    scores: np.ndarray,
    rule_scores: np.ndarray,
) -> None:
    test = report["test"]
    tags = {"stage": STAGE, "model_name": MODEL_NAME, "model_version": str(version.version)}
    with start_run(f"test evaluation v{version.version}", tags=tags) as run:
        mlflow.log_params(
            {
                "threshold": decisions["policy"].threshold,
                "capacity_fraction": CONTACT_FRACTION,
                "calibration_tolerance": CALIBRATION_TOLERANCE,
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test["champion"].items()})
        mlflow.log_metrics({f"test_b2_{k}": v for k, v in test["b2_rule"].items()})
        for name, c in test["vs_b2"].items():
            mlflow.log_metrics({f"vs_b2_{name}_diff": c["difference"], f"vs_b2_{name}_ci_low": c["ci_low"]})
        mlflow.log_metrics({f"policy_{k}": float(v) for k, v in test["policy"].items()})
        mlflow.log_metrics(
            {"test_ece": test["calibration"]["ece"], "oof_ece": decisions["ece"], **test["fairness"]}
        )
        mlflow.set_tag("passed_success_criteria", str(test["passed"]).lower())

        mlflow.log_dict(report, "evaluation.json")
        mlflow.log_text(segments.round(4).to_csv(index=False), "segments.csv")
        mlflow.log_text(decisions["calibration_table"].round(4).to_csv(index=False), "calibration_oof.csv")
        mlflow.log_text(importance.round(4).to_csv(), "shap_importance.csv")
        figures = {
            "pr_curve.png": plots.pr_curves(
                y_test, {"champion": scores, "B2 contract+tenure rule": rule_scores}
            ),
            "calibration_test.png": plots.reliability(
                calibration_table(y_test, scores), "Calibration on the test set"
            ),
            "profit_by_contacts.png": plots.profit_by_contacts(
                test_curve, CONTACT_FRACTION, "Expected profit vs customers contacted (test)"
            ),
            "shap_importance.png": plots.importance(importance, "What drives the churn score (test set)"),
        }
        for name, fig in figures.items():
            mlflow.log_figure(fig, name)
            plt.close(fig)

    client = MlflowClient()
    version_tags = {
        "evaluation_run_id": run.info.run_id,
        "passed_success_criteria": str(test["passed"]).lower(),
        "decision_threshold": str(decisions["policy"].threshold),
        "test_pr_auc": f"{test['champion']['pr_auc']:.4f}",
        "test_profit_at_k": f"{test['champion']['profit_at_k']:.0f}",
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(MODEL_NAME, str(version.version), key, value)
    report["mlflow_run_id"] = run.info.run_id


def _print(report: dict[str, Any]) -> None:
    d, t = report["decisions_on_training_oof"], report["test"]
    champion, rule = t["champion"], t["b2_rule"]
    print(f"\n{report['model']['name']} v{report['model']['version']} ({report['model']['family']})")
    print(
        f"\nDecisions (training out-of-fold): ECE {d['ece']:.3f}, calibrated enough: {d['calibrated_enough']}"
    )
    print(
        f"  threshold {d['policy']['threshold']:.2f} (break-even {d['break_even_probability']:.3f}), "
        f"capacity {d['policy']['capacity_fraction']:.0%}; unlimited capacity would contact "
        f"{d['uncapped_best_share']:.0%}"
    )
    print(f"\nTest set: {t['customers']} customers, churn rate {t['churn_rate']:.3f}")
    print(f"  {'metric':<16}{'champion':>12}{'B2 rule':>12}{'difference [95% CI]':>34}")
    for name in ("pr_auc", "roc_auc", "brier", "precision_at_k", "recall_at_k", "profit_at_k"):
        line = f"  {name:<16}{champion[name]:>12.3f}{rule[name]:>12.3f}"
        if name in t["vs_b2"]:
            c = t["vs_b2"][name]
            line += f"{c['difference']:>+14.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
        print(line)
    print(f"  success criteria: {t['success_criteria']} -> passed: {t['passed']}")
    p = t["policy"]
    print(
        f"\nPolicy on test: contact {p['contacted']} ({p['contacted_share']:.0%}), "
        f"precision {p['precision']:.2f}, "
        f"recall {p['recall']:.2f}, profit ${p['profit']:,.0f} (${p['profit_per_contact']:.0f}/contact)"
    )
    print(f"  calibration on test: ECE {t['calibration']['ece']:.3f}; fairness recall gaps: {t['fairness']}")
    top = list(report["explanations"]["global_importance"].items())[:5]
    print("\nTop drivers (mean |SHAP|): " + ", ".join(f"{k} {v:.2f}" for k, v in top))
    for example in report["explanations"]["top_flagged_examples"][:3]:
        print(f"  #{example['rank']} P(leaves)={example['p_leaves']}: " + "; ".join(example["reasons"]))
    print(f"\nLogged to MLflow run {report['mlflow_run_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the champion once on the test set.")
    parser.add_argument("--force", action="store_true", help="re-run an evaluation that already exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    configure()
    try:
        report = run(force=args.force)
    except AlreadyEvaluated as error:
        logger.error("%s", error)
        return 2
    _print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
