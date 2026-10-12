"""The weekly call list: score every active customer with the champion and pick who to call.

    fetch -> load & validate active customers -> score (same code as the API) -> rank -> top N
          -> data/outputs/call_lists/call_list_<date>.csv  + a summary in Prefect (no customer IDs)

Who is on the list: customers at or above the champion's call threshold, highest risk first, at most
CALLS_PER_WEEK. Each row comes with its top reasons, so the team knows what to talk about, and its
expected value, which is meaningful because the probabilities are calibrated (module 5).

Run once:  uv run python -m retention.pipeline.score
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from retention.config import BUSINESS, CALLS_PER_WEEK, ID_COLUMN, TARGET
from retention.data.clean import clean, read_raw
from retention.data.download import PROJECT_ROOT, RAW_PATH, download
from retention.serving.service import ScoringService
from retention.tracking import connect

OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "call_lists"


@task(retries=3, retry_delay_seconds=[10, 30, 60])
def fetch_customers() -> None:
    download()


@task
def load_active_customers() -> pd.DataFrame:
    """Validated customers who are still with us.

    Dataset limitation: the IBM sample is one snapshot, so "active" means Churn = No in it, and these
    customers were part of the training data. In production this reads the current customer table.
    """
    customers = clean(read_raw(RAW_PATH))  # the same data contract as training
    return customers[customers[TARGET] == 0].drop(columns=[TARGET]).reset_index(drop=True)


@task
def score_customers(customers: pd.DataFrame) -> tuple[pd.DataFrame, str, float]:
    connect()
    service = ScoringService.from_registry()
    scored = pd.concat([customers[[ID_COLUMN]], service.score(customers)], axis=1)
    return scored, service.version, service.threshold


def expected_value(p: pd.Series) -> pd.Series:
    """Expected profit of calling a customer with churn probability p (calibrated)."""
    return p * BUSINESS.save_rate * BUSINESS.value_of_saving - BUSINESS.cost_per_contact


@task
def build_call_list(scored: pd.DataFrame, calls: int, version: str, scored_at: str) -> pd.DataFrame:
    eligible = scored[scored["worth_calling"]].sort_values("p_leaves", ascending=False, kind="stable")
    call_list = eligible.head(calls).reset_index(drop=True)
    return pd.DataFrame(
        {
            "rank": range(1, len(call_list) + 1),
            ID_COLUMN: call_list[ID_COLUMN],
            "p_leaves": call_list["p_leaves"],
            "expected_value": expected_value(call_list["p_leaves"]).round(2),
            "reasons": call_list["reasons"].map("; ".join),
            "model_version": version,
            "scored_at": scored_at,
        }
    )


@task
def write_call_list(call_list: pd.DataFrame, scored_at: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"call_list_{scored_at[:10]}.csv"
    call_list.to_csv(path, index=False)
    return path


@flow(name="weekly-call-list", log_prints=True)
def weekly_call_list(calls: int = CALLS_PER_WEEK) -> dict[str, Any]:
    logger = get_run_logger()
    scored_at = datetime.now(UTC).isoformat(timespec="seconds")
    fetch_customers()
    customers = load_active_customers()
    scored, version, threshold = score_customers(customers)
    call_list = build_call_list(scored, calls, version, scored_at)
    path = write_call_list(call_list, scored_at, OUTPUT_DIR)

    worth_calling = int(scored["worth_calling"].sum())
    summary = {
        "model_version": version,
        "threshold": threshold,
        "customers_scored": len(scored),
        "worth_calling": worth_calling,
        "on_the_list": len(call_list),
        "expected_saves": round(float((call_list["p_leaves"] * BUSINESS.save_rate).sum()), 1),
        "expected_value": round(float(call_list["expected_value"].sum()), 2),
        "file": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
    }
    if worth_calling > calls:
        logger.info(
            "%d customers are worth calling but capacity is %d: the list is capped (see docs/evaluation.md)",
            worth_calling,
            calls,
        )
    lines = ["## Weekly call list", ""] + [f"- **{k}:** {v}" for k, v in summary.items()]
    create_markdown_artifact(
        "\n".join(lines), key="weekly-call-list", description="Counts only, no customer IDs"
    )
    return summary


def main() -> int:
    summary = weekly_call_list()
    print({k: v for k, v in summary.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
