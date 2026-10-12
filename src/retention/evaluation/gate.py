"""The promotion gate: does a challenger replace the champion? (criteria version 2, problem_framing.md)

1. The challenger must meet the shipping criteria against the B2 rule (from its test evaluation).
2. If there is a champion, the challenger must not be worse than it on PR-AUC: the 95% paired bootstrap
   interval of (challenger - champion) must have its lower bound above -PR_AUC_NON_INFERIORITY_MARGIN.

"Not worse", not "better": a retrained model on fresh data should replace a stale one even if it only ties.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlflow.sklearn
import pandas as pd
from sklearn.metrics import average_precision_score

from retention.config import PR_AUC_NON_INFERIORITY_MARGIN, TARGET
from retention.data.prepare import PROCESSED_DIR
from retention.evaluation.bootstrap import Comparison, paired_bootstrap
from retention.models.champion import MODEL_NAME, model_input


@dataclass(frozen=True)
class Decision:
    promote: bool
    reasons: list[str] = field(default_factory=list)


def compare_versions(challenger: str, champion: str, name: str = MODEL_NAME) -> Comparison:
    """PR-AUC of challenger vs champion on the test set, with a paired 95% interval."""
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X, y = model_input(test), test[TARGET].to_numpy()
    scores = {
        version: mlflow.sklearn.load_model(f"models:/{name}/{version}").predict_proba(X)[:, 1]
        for version in (challenger, champion)
    }
    return paired_bootstrap(y, scores[challenger], scores[champion], {"pr_auc": average_precision_score})[
        "pr_auc"
    ]


def decide(passed_criteria: bool, vs_champion: Comparison | None) -> Decision:
    reasons = [
        "meets the shipping criteria against the B2 rule"
        if passed_criteria
        else "does not meet the shipping criteria against the B2 rule"
    ]
    not_worse = vs_champion is None or vs_champion.ci_low > -PR_AUC_NON_INFERIORITY_MARGIN
    if vs_champion is None:
        reasons.append("no current champion")
    else:
        verdict = "not worse than" if not_worse else "worse than"
        reasons.append(
            f"{verdict} the champion on PR-AUC: {vs_champion.difference:+.4f} "
            f"[{vs_champion.ci_low:+.4f}, {vs_champion.ci_high:+.4f}], margin {PR_AUC_NON_INFERIORITY_MARGIN}"
        )
    return Decision(promote=passed_criteria and not_worse, reasons=reasons)
