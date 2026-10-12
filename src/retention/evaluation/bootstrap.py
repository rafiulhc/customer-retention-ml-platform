"""Paired bootstrap: how sure are we that the model beats the baseline on THIS test set?

The test set is one sample of customers; a different sample would give different numbers. Resampling
customers with replacement (the same resample for both models, so the comparison stays paired)
shows how much the difference could move, as a 95% interval.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from retention.models.metrics import Array

Metric = Callable[[Array, Array], float]


@dataclass(frozen=True)
class Comparison:
    model: float
    baseline: float
    difference: float
    ci_low: float  # 95% interval of the difference
    ci_high: float
    prob_not_better: float  # share of resamples where the model does not beat the baseline

    @property
    def clearly_better(self) -> bool:
        """The whole 95% interval is above zero."""
        return self.ci_low > 0


def paired_bootstrap(
    y_true: Array,
    model_scores: Array,
    baseline_scores: Array,
    metrics: Mapping[str, Metric],
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Comparison]:
    y = np.asarray(y_true, dtype=np.int64)
    a = np.asarray(model_scores, dtype=np.float64)
    b = np.asarray(baseline_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out: dict[str, Comparison] = {}
    samples = [rng.integers(0, len(y), len(y)) for _ in range(n_resamples)]
    for name, metric in metrics.items():
        diffs = np.array([metric(y[i], a[i]) - metric(y[i], b[i]) for i in samples])
        low, high = np.percentile(diffs, [2.5, 97.5])
        model_value, baseline_value = metric(y, a), metric(y, b)
        out[name] = Comparison(
            model=model_value,
            baseline=baseline_value,
            difference=model_value - baseline_value,
            ci_low=float(low),
            ci_high=float(high),
            prob_not_better=float((diffs <= 0).mean()),
        )
    return out
