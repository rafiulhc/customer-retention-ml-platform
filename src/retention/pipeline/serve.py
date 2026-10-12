"""Run the pipelines on their schedules: one long-running process that the Prefect server triggers.

    uv run python -m retention.pipeline.serve

- monthly-retrain:  06:00 UTC on the 1st ("retrain monthly", docs/problem_framing.md; drift-triggered
  runs come in module 10)
- weekly-call-list: 05:00 UTC every Monday, so the list is ready when the team starts the week

Runs can also be started from the Prefect UI. Stop with Ctrl+C. In the cloud (module 9) the same flows
run on managed infrastructure instead of a terminal.
"""

from __future__ import annotations

from prefect import serve
from prefect.deployments.runner import RunnerDeployment

from retention.pipeline.flows import retrain
from retention.pipeline.score import weekly_call_list

MONTHLY = "0 6 1 * *"
WEEKLY = "0 5 * * 1"


def _sync(deployment: object) -> RunnerDeployment:
    """`to_deployment` returns the deployment directly when called outside async code."""
    assert isinstance(deployment, RunnerDeployment)
    return deployment


def main() -> None:
    serve(
        _sync(
            retrain.to_deployment(
                name="monthly-retrain",
                cron=MONTHLY,
                tags=["retention", "training"],
                description="Retrain on fresh data; promote only if the gate passes (criteria version 2).",
                parameters={"force": False, "auto_promote": True},
            )
        ),
        _sync(
            weekly_call_list.to_deployment(
                name="weekly-call-list",
                cron=WEEKLY,
                tags=["retention", "scoring"],
                description="Score active customers with @champion and write the week's call list.",
            )
        ),
    )


if __name__ == "__main__":
    main()
