"""Evaluation figures, returned as matplotlib Figures so they can be logged to MLflow or saved."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed: figures are rendered to files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from retention.models.metrics import Array


def pr_curves(y_true: Array, curves: dict[str, Array]) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, scores in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=name)
    ax.axhline(float(np.mean(y_true)), color="grey", linestyle=":", label="base rate")
    ax.set(xlabel="Recall (share of churners found)", ylabel="Precision (share of calls that matter)")
    ax.set_title("Precision-recall on the test set")
    ax.legend()
    fig.tight_layout()
    return fig


def reliability(table: pd.DataFrame, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="perfectly calibrated")
    ax.plot(table["predicted"], table["observed"], marker="o", label="model")
    ax.set(xlabel="Predicted P(leaves)", ylabel="Observed churn rate", xlim=(0, 1), ylim=(0, 1), title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def profit_by_contacts(curve: pd.DataFrame, capacity_share: float, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(curve["share"] * 100, curve["profit"])
    ax.axvline(capacity_share * 100, color="grey", linestyle=":", label=f"capacity ({capacity_share:.0%})")
    best = int(np.argmax(curve["profit"].to_numpy()))
    share, value = float(curve["share"].iloc[best]), float(curve["profit"].iloc[best])
    ax.scatter([share * 100], [value], zorder=3, label=f"max at {share:.0%} contacted")
    ax.set(xlabel="Customers contacted, highest risk first (%)", ylabel="Expected profit ($)", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def importance(values: pd.Series, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    values.iloc[::-1].plot.barh(ax=ax)
    ax.set(xlabel="Mean |SHAP value| (log-odds)", title=title)
    fig.tight_layout()
    return fig
