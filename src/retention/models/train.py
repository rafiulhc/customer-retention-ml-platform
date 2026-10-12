"""Compare every candidate with stratified cross-validation on the training set (the test set stays locked).

Run:  uv run python -m retention.models.train
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from retention.config import CV_FOLDS, SEED, TARGET
from retention.data.download import PROJECT_ROOT
from retention.data.prepare import PROCESSED_DIR
from retention.models.baselines import ContractRule, ContractTenureRule, NobodyLeaves
from retention.models.features import model_pipeline
from retention.models.metrics import evaluate

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
PRIMARY_METRIC = "pr_auc"
REFERENCE = "B2 contract+tenure"  # the rule a model must beat (docs/problem_framing.md)


def candidates() -> dict[str, Callable[[], Any]]:
    """Every model to compare. Factories, so each fold gets a fresh, unfitted model."""
    return {
        "B0 nobody leaves": NobodyLeaves,
        "B1 contract": ContractRule,
        "B2 contract+tenure": ContractTenureRule,
        "B3 logistic regression": lambda: model_pipeline(
            LogisticRegression(max_iter=2000), scale_numeric=True
        ),
        "XGBoost": lambda: model_pipeline(
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=SEED,
            ),
            scale_numeric=False,
        ),
        "LightGBM": lambda: model_pipeline(
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=SEED,
                verbose=-1,
            ),
            scale_numeric=False,
        ),
    }


FINAL_MODEL = "B3 logistic regression"  # chosen in docs/model_selection.md


def final_model() -> Any:
    """The model chosen in docs/model_selection.md: a fresh, unfitted pipeline.

    Every later stage (tracking, evaluation, serving) builds the model through this function, so the
    choice lives in one place.
    """
    return candidates()[FINAL_MODEL]()


def cross_validate(
    df: pd.DataFrame, models: dict[str, Callable[[], Any]], folds: int = CV_FOLDS, seed: int = SEED
) -> pd.DataFrame:
    """One row per (model, fold). Every model sees exactly the same folds, so comparisons are paired."""
    y = df[TARGET].to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(df, y)):
        train, valid = df.iloc[train_idx], df.iloc[valid_idx]
        for name, make in models.items():
            start = time.perf_counter()
            model = clone(make()).fit(train, train[TARGET])
            scores = model.predict_proba(valid)[:, 1]
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "fit_seconds": time.perf_counter() - start,
                    **evaluate(valid[TARGET].to_numpy(), scores),
                }
            )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation per model, best primary metric first."""
    metrics = results.drop(columns=["model", "fold"])
    summary = metrics.groupby(results["model"]).agg(["mean", "std"])
    return summary.sort_values((PRIMARY_METRIC, "mean"), ascending=False)


def paired_comparison(results: pd.DataFrame, model: str, reference: str = REFERENCE) -> dict[str, Any]:
    """How a model does against the reference on the SAME folds: mean difference and folds won."""
    out: dict[str, Any] = {}
    for metric in (PRIMARY_METRIC, "profit_at_k"):
        a = results[results["model"] == model].sort_values("fold")[metric].to_numpy()
        b = results[results["model"] == reference].sort_values("fold")[metric].to_numpy()
        diff = a - b
        out[metric] = {
            "mean_difference": float(diff.mean()),
            "std_difference": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "folds_won": int((diff > 0).sum()),
            "folds": len(diff),
        }
    return out


def format_table(summary: pd.DataFrame) -> str:
    cols = [PRIMARY_METRIC, "roc_auc", "brier", "precision_at_k", "profit_at_k", "fit_seconds"]
    lines = [f"{'model':<24}" + "".join(f"{c:>20}" for c in cols)]
    for model, row in summary.iterrows():
        cells = []
        for c in cols:
            mean, std = row[(c, "mean")], row[(c, "std")]
            cells.append(
                f"{mean:>12,.0f} ±{std:>6,.0f}" if c == "profit_at_k" else f"{mean:>12.3f} ±{std:>5.3f}"
            )
        lines.append(f"{model!s:<24}" + "".join(f"{cell:>20}" for cell in cells))
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    logger.info(
        "Cross-validating %d models on %d training customers (%d folds)",
        len(candidates()),
        len(train),
        CV_FOLDS,
    )

    results = cross_validate(train, candidates())
    summary = summarise(results)
    best = str(summary.index[0])
    comparison = paired_comparison(results, best)

    print("\n" + format_table(summary) + "\n")
    for metric, c in comparison.items():
        print(
            f"{best} vs {REFERENCE} on {metric}: {c['mean_difference']:+,.3f} "
            f"(± {c['std_difference']:,.3f}), better on {c['folds_won']}/{c['folds']} folds"
        )

    REPORTS_DIR.mkdir(exist_ok=True)
    report = {
        "primary_metric": PRIMARY_METRIC,
        "folds": CV_FOLDS,
        "seed": SEED,
        "best_model": best,
        "vs_reference": {"reference": REFERENCE, **comparison},
        "per_fold": results.round(4).to_dict(orient="records"),
    }
    (REPORTS_DIR / "cv_results.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    logger.info("Report written to %s", REPORTS_DIR / "cv_results.json")
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
