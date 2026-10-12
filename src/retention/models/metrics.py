"""Ranking metrics under a capacity limit: the team can only contact the top fraction of customers."""

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from retention.config import BUSINESS, CONTACT_FRACTION, BusinessAssumptions

Array = npt.NDArray[np.float64] | npt.NDArray[np.int64]


def _check(y_true: Array, scores: Array) -> None:
    if len(y_true) != len(scores):
        raise ValueError(f"y_true has {len(y_true)} values but scores has {len(scores)}")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")


def top_k_mask(scores: Array, fraction: float = CONTACT_FRACTION) -> npt.NDArray[np.bool_]:
    """True for the top `fraction` of customers by score (always at least one).

    Ties are broken by original order, so the result is deterministic.
    """
    k = max(1, round(len(scores) * fraction))
    # highest score first; "stable" keeps tied customers in their original order
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def precision_at_k(y_true: Array, scores: Array, fraction: float = CONTACT_FRACTION) -> float:
    """Of the customers we contact, the share who really churn."""
    _check(y_true, scores)
    return float(np.asarray(y_true)[top_k_mask(scores, fraction)].mean())


def recall_at_k(y_true: Array, scores: Array, fraction: float = CONTACT_FRACTION) -> float:
    """Of all churners, the share we reach."""
    _check(y_true, scores)
    y = np.asarray(y_true)
    churners = y.sum()
    return float(y[top_k_mask(scores, fraction)].sum() / churners) if churners else 0.0


def expected_profit_at_k(
    y_true: Array, scores: Array, fraction: float = CONTACT_FRACTION, business: BusinessAssumptions = BUSINESS
) -> float:
    """Expected value of contacting the top fraction: useful contacts earn, wasted ones cost."""
    _check(y_true, scores)
    contacted = np.asarray(y_true)[top_k_mask(scores, fraction)]
    true_positives = int(contacted.sum())
    false_positives = len(contacted) - true_positives
    return true_positives * business.true_positive_value + false_positives * business.false_positive_value


def evaluate(y_true: Array, scores: Array, fraction: float = CONTACT_FRACTION) -> dict[str, float]:
    """Every metric from the framing document, for one set of predictions."""
    _check(y_true, scores)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
        "precision_at_k": precision_at_k(y_true, scores, fraction),
        "recall_at_k": recall_at_k(y_true, scores, fraction),
        "profit_at_k": expected_profit_at_k(y_true, scores, fraction),
    }
