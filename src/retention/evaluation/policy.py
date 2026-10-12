"""The decision policy: who gets a retention call.

Rank customers by P(leaves); contact those above a probability threshold, up to the team's capacity.
The threshold is chosen on out-of-fold predictions to maximise expected profit, and compared with
the break-even probability from the business assumptions (they agree only if the model is calibrated).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from retention.config import BUSINESS, CONTACT_FRACTION, BusinessAssumptions
from retention.models.metrics import Array


@dataclass(frozen=True)
class Policy:
    threshold: float  # contact only if P(leaves) >= threshold
    capacity_fraction: float = CONTACT_FRACTION  # and at most this share of customers

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def contact_mask(scores: Array, policy: Policy) -> npt.NDArray[np.bool_]:
    """Who is contacted: the highest scores first, above the threshold, within capacity."""
    s = np.asarray(scores, dtype=float)
    capacity = max(1, round(len(s) * policy.capacity_fraction))
    order = np.argsort(-s, kind="stable")[:capacity]
    mask = np.zeros(len(s), dtype=bool)
    mask[order[s[order] >= policy.threshold]] = True
    return mask


def profit(
    y_true: Array, contacted: npt.NDArray[np.bool_], business: BusinessAssumptions = BUSINESS
) -> float:
    """Expected value of a contact list: each true positive earns, each false positive costs."""
    hits = int(np.asarray(y_true)[contacted].sum())
    misses = int(contacted.sum()) - hits
    return hits * business.true_positive_value + misses * business.false_positive_value


def profit_curve(y_true: Array, scores: Array, business: BusinessAssumptions = BUSINESS) -> pd.DataFrame:
    """Profit of contacting the top n customers, for every n (no capacity limit)."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    y = np.asarray(y_true)[order]
    hits = np.cumsum(y)
    contacts = np.arange(1, len(y) + 1)
    value = hits * business.true_positive_value + (contacts - hits) * business.false_positive_value
    return pd.DataFrame(
        {
            "contacts": contacts,
            "share": contacts / len(y),
            "threshold": np.asarray(scores)[order],
            "profit": value,
        }
    )


def choose_threshold(
    y_true: Array,
    scores: Array,
    capacity_fraction: float = CONTACT_FRACTION,
    grid: npt.NDArray[np.float64] | None = None,
) -> Policy:
    """The threshold with the highest profit within capacity; on ties, the highest (fewest calls)."""
    candidates = np.round(np.arange(0.0, 1.0001, 0.01), 2) if grid is None else grid
    best_threshold, best_profit = 1.0, -np.inf
    for t in candidates[::-1]:  # high to low, so a tie keeps the higher threshold
        value = profit(y_true, contact_mask(scores, Policy(float(t), capacity_fraction)))
        if value > best_profit:
            best_threshold, best_profit = float(t), value
    return Policy(threshold=best_threshold, capacity_fraction=capacity_fraction)
