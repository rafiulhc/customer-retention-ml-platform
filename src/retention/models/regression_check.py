"""The CI model check: did this code change make the model worse? Cross-validation on training data only.

CI runs on every push, so it must never look at the test set: repeated looks would slowly tune the code to
those 1,409 customers. The test set stays with the promotion gate (once per model version, module 6).

The check retrains the chosen model and the B2 rule on the same seeded folds as module 3 and compares with
`ci/model_baseline.json`. The code is deterministic, so an unchanged model reproduces the baseline exactly.
A change that improves the model beyond the tolerance passes, with a reminder to update the baseline in the
same pull request, so the bar moves up only through review.

    uv run python -m retention.models.regression_check                      # check
    uv run python -m retention.models.regression_check --update-baseline    # after an intended improvement
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from retention.config import CV_FOLDS, SEED
from retention.data.download import PROJECT_ROOT
from retention.data.prepare import PROCESSED_DIR
from retention.lineage import content_sha256, git_commit
from retention.models.champion import Check
from retention.models.train import (
    FINAL_MODEL,
    REFERENCE,
    candidates,
    cross_validate,
    paired_comparison,
    summarise,
)

BASELINE_PATH = PROJECT_ROOT / "ci" / "model_baseline.json"
PR_AUC_TOLERANCE = 0.005  # absolute: small enough to catch a real regression, above rounding noise
PROFIT_TOLERANCE = 0.03  # relative


def measure(train: pd.DataFrame) -> dict[str, Any]:
    models = {name: make for name, make in candidates().items() if name in (FINAL_MODEL, REFERENCE)}
    results = cross_validate(train, models)
    summary = summarise(results)
    vs_rule = paired_comparison(results, FINAL_MODEL, reference=REFERENCE)

    def stat(metric: str, which: str) -> float:
        return float(summary[(metric, which)].to_numpy()[list(summary.index).index(FINAL_MODEL)])

    return {
        "model": FINAL_MODEL,
        "folds": CV_FOLDS,
        "seed": SEED,
        "data_fingerprint": content_sha256(train),
        "pr_auc_mean": round(stat("pr_auc", "mean"), 4),
        "pr_auc_std": round(stat("pr_auc", "std"), 4),
        "profit_at_k_mean": round(stat("profit_at_k", "mean"), 2),
        "folds_beating_b2_pr_auc": vs_rule["pr_auc"]["folds_won"],
        "folds_beating_b2_profit": vs_rule["profit_at_k"]["folds_won"],
    }


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> list[Check]:
    pr_floor = baseline["pr_auc_mean"] - PR_AUC_TOLERANCE
    profit_floor = baseline["profit_at_k_mean"] * (1 - PROFIT_TOLERANCE)
    folds = current["folds"]
    checks = [
        Check(
            "same training data as the baseline",
            current["data_fingerprint"] == baseline["data_fingerprint"],
            "comparable" if current["data_fingerprint"] == baseline["data_fingerprint"] else "data changed",
        ),
        Check(
            "PR-AUC not below the baseline",
            current["pr_auc_mean"] >= pr_floor,
            f"{current['pr_auc_mean']:.4f} vs baseline {baseline['pr_auc_mean']:.4f} (floor {pr_floor:.4f})",
        ),
        Check(
            "profit@10% not below the baseline",
            current["profit_at_k_mean"] >= profit_floor,
            f"${current['profit_at_k_mean']:,.0f} vs baseline ${baseline['profit_at_k_mean']:,.0f}",
        ),
        Check(
            "beats the B2 rule on every fold",
            current["folds_beating_b2_pr_auc"] == folds and current["folds_beating_b2_profit"] == folds,
            f"PR-AUC {current['folds_beating_b2_pr_auc']}/{folds}, "
            f"profit {current['folds_beating_b2_profit']}/{folds}",
        ),
    ]
    return checks


def report(checks: list[Check], current: dict[str, Any], baseline: dict[str, Any]) -> str:
    lines = ["## Model check (cross-validation on training data, never the test set)", ""]
    lines += ["| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c.name} | {'✅ pass' if c.passed else '❌ fail'} | {c.detail} |" for c in checks]
    if current["pr_auc_mean"] > baseline["pr_auc_mean"] + PR_AUC_TOLERANCE:
        lines += [
            "",
            "**Improved beyond the tolerance:** update the baseline in this pull request "
            "(`uv run python -m retention.models.regression_check --update-baseline`).",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="CI model check against the committed baseline.")
    parser.add_argument(
        "--update-baseline", action="store_true", help="write the current numbers as the baseline"
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    args = parser.parse_args()

    current = measure(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps({**current, "set_at_commit": git_commit()}, indent=2) + "\n")
        print(f"Baseline written to {args.baseline}")
        return 0

    baseline = json.loads(args.baseline.read_text())
    checks = compare(current, baseline)
    text = report(checks, current, baseline)
    print(text)
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):  # shown on the run's page in GitHub
        Path(summary_file).write_text(text + "\n")
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
