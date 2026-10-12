"""The retraining pipeline: new data in, a verified and evaluated model out, promoted only if it earns it.

    fetch -> prepare & validate -> new data? -> compare models -> register challenger -> verify
          -> evaluate once on the test set -> gate -> promote (or keep the current champion)

Each step is a Prefect task: the run, its logs, timings and failures are visible in the Prefect UI
(http://localhost:4200), and the network step retries on its own. Every model artifact and number still
goes to MLflow; Prefect records *what ran and how it went*, MLflow records *what was produced*.

Run once:   uv run python -m retention.pipeline.flows [--force]
Scheduled:  uv run python -m retention.pipeline.serve
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd
from mlflow import MlflowClient
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from retention.data.download import download
from retention.data.prepare import PROCESSED_DIR, prepare
from retention.evaluation import evaluate, gate
from retention.lineage import same_training_data
from retention.models.champion import (
    CHAMPION,
    MODEL_NAME,
    current_champion,
    load_version,
    model_input,
    verify,
)
from retention.models.register import register
from retention.models.train import FINAL_MODEL, model_selection
from retention.tracking import configure


@task(retries=3, retry_delay_seconds=[10, 30, 60])
def fetch_data() -> str:
    """Download the raw data (checksum-verified). Network failures are retried."""
    return str(download())


@task
def prepare_data() -> dict[str, Any]:
    """Validate (pandera), clean and split: the manifest holds the new data's checksums."""
    return prepare()


@task
def needs_training(manifest: dict[str, Any], force: bool) -> tuple[bool, str]:
    """Train only when the data differs from what the champion was trained on (or when forced)."""
    champion = current_champion(MlflowClient())
    if champion is None:
        return True, "no champion yet"
    tags = MlflowClient().get_model_version(MODEL_NAME, champion).tags
    if not same_training_data(tags, PROCESSED_DIR / manifest["outputs"]["train"]["file"]):
        return True, "the training data changed since the champion was trained"
    if force:
        return True, "forced (the data is unchanged)"
    return False, f"the champion (v{champion}) was trained on exactly this data"


@task
def compare_models() -> dict[str, Any]:
    """Re-run the model comparison so a change in the ranking is visible (logged to MLflow)."""
    run_id, _, summary = model_selection(pd.read_parquet(PROCESSED_DIR / "train.parquet"))
    ranking = [str(name) for name in summary.index]
    if ranking[0] != FINAL_MODEL:
        get_run_logger().warning(
            "%s now ranks first (the chosen family, %s, ranks %d). Revisit docs/model_selection.md.",
            ranking[0],
            FINAL_MODEL,
            ranking.index(FINAL_MODEL) + 1,
        )
    return {"run_id": run_id, "ranking": ranking}


@task
def register_challenger() -> str:
    """Train the chosen family on the new training set and register it, without promoting it."""
    result = register(
        pd.read_parquet(PROCESSED_DIR / "train.parquet"), promote=False, first_becomes_champion=False
    )
    return result.version


@task
def verify_challenger(version: str) -> None:
    """The contract checks serving relies on. A broken artifact fails the run here, before evaluation."""
    sample = model_input(pd.read_parquet(PROCESSED_DIR / "train.parquet").head(200))
    failed = [c for c in verify(load_version(version), sample) if not c.passed]
    if failed:
        raise RuntimeError(
            f"v{version} failed verification: " + "; ".join(f"{c.name} ({c.detail})" for c in failed)
        )


@task
def evaluate_challenger(version: str) -> dict[str, Any]:
    """The challenger's one look at the test set."""
    return evaluate.run(version_number=version)


@task
def decide(version: str, report: dict[str, Any]) -> gate.Decision:
    champion = current_champion(MlflowClient())
    vs_champion = gate.compare_versions(version, champion) if champion and champion != version else None
    return gate.decide(report["test"]["passed"], vs_champion)


@task
def promote(version: str, decision: gate.Decision) -> None:
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    client.set_model_version_tag(
        MODEL_NAME, version, "promoted_by", "pipeline gate: " + "; ".join(decision.reasons)
    )


def _summary(outcome: dict[str, Any]) -> str:
    lines = ["## Retraining run", "", f"**Outcome:** {outcome['outcome']}", ""]
    for key, value in outcome.items():
        if key != "outcome":
            lines.append(f"- **{key}:** {value}")
    return "\n".join(lines)


@flow(name="retrain", log_prints=True)
def retrain(force: bool = False, auto_promote: bool = True) -> dict[str, Any]:
    """Retrain on fresh data and promote the result if it passes the gate."""
    logger = get_run_logger()
    configure()

    fetch_data()
    manifest = prepare_data()
    train_needed, why = needs_training(manifest, force)
    if not train_needed:
        outcome: dict[str, Any] = {"outcome": "skipped", "reason": why}
        logger.info("Nothing to do: %s", why)
    else:
        logger.info("Training: %s", why)
        selection = compare_models()
        version = register_challenger()
        verify_challenger(version)
        report = evaluate_challenger(version)
        decision = decide(version, report)
        champion_before = current_champion(MlflowClient())
        if decision.promote and auto_promote:
            promote(version, decision)
        outcome = {
            "outcome": f"v{version} promoted to @{CHAMPION}"
            if decision.promote and auto_promote
            else f"v{version} registered, not promoted",
            "reason": why,
            "champion_before": f"v{champion_before}",
            "gate": " / ".join(decision.reasons),
            "test_pr_auc": round(report["test"]["champion"]["pr_auc"], 4),
            "model_selection_run": selection["run_id"],
            "evaluation_run": report["mlflow_run_id"],
        }
    create_markdown_artifact(
        _summary(outcome), key="retrain-summary", description="What this run did and why"
    )
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the retraining pipeline once.")
    parser.add_argument("--force", action="store_true", help="retrain even if the data is unchanged")
    parser.add_argument("--no-promote", action="store_true", help="register and evaluate, but never promote")
    args = parser.parse_args()
    outcome = retrain(force=args.force, auto_promote=not args.no_promote)
    print(outcome["outcome"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
