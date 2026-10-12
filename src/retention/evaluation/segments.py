"""Model quality per customer group: where it works, where it doesn't, and whether it is fair.

A retention call comes with a discount, so being contacted is a benefit. The fairness question is
therefore "do churners in every group get the same chance of being reached?" (equal opportunity:
recall per group), not "is everyone flagged at the same rate?" (groups really do churn at different rates).
`gender` and `senior_citizen` are not model features, but they are reported here so gaps are visible.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import average_precision_score

from retention.config import EXCLUDED_FOR_FAIRNESS

SEGMENT_COLUMNS = [*EXCLUDED_FOR_FAIRNESS, "contract", "internet_service", "tenure_band"]
TENURE_BANDS = (-1, 12, 24, 48, np.inf)
TENURE_LABELS = ("0-12 months", "13-24 months", "25-48 months", "49+ months")


def with_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tenure_band"] = pd.cut(out["tenure"], TENURE_BANDS, labels=TENURE_LABELS).astype(str)
    return out


def segment_report(
    df: pd.DataFrame,
    y_true: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    contacted: npt.NDArray[np.bool_],
    columns: list[str] = SEGMENT_COLUMNS,
) -> pd.DataFrame:
    """One row per (column, group). Small groups are kept but flagged: their numbers are noisy."""
    frame = with_tenure_band(df).reset_index(drop=True)
    frame["_y"], frame["_p"], frame["_contacted"] = y_true, scores, contacted
    rows = []
    for column in columns:
        for group, part in frame.groupby(column, observed=True):
            churners = part["_y"].sum()
            reached = part.loc[part["_contacted"], "_y"]
            rows.append(
                {
                    "segment": column,
                    "group": str(group),
                    "customers": len(part),
                    "churn_rate": part["_y"].mean(),
                    "mean_predicted": part["_p"].mean(),
                    "pr_auc": average_precision_score(part["_y"], part["_p"])
                    if 0 < churners < len(part)
                    else np.nan,
                    "contacted_share": part["_contacted"].mean(),
                    "recall": reached.sum() / churners if churners else np.nan,
                    "precision": reached.mean() if len(reached) else np.nan,
                    "small_group": len(part) < 100,
                }
            )
    return pd.DataFrame(rows)


def largest_gap(report: pd.DataFrame, segment: str, metric: str = "recall") -> float:
    """Spread of a metric between the best and worst group of a segment (0 = equal)."""
    values = report.loc[report["segment"] == segment, metric].dropna()
    return float(values.max() - values.min()) if len(values) else 0.0
