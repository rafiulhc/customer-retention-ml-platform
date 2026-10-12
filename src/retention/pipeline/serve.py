"""Run the retraining pipeline on a schedule: a long-running process that the Prefect server triggers.

    uv run python -m retention.pipeline.serve

Monthly (06:00 UTC on the 1st), as set in docs/problem_framing.md ("retrain monthly, or when drift is
detected"; drift-triggered runs come in module 10). Runs can also be started from the Prefect UI.
Stop with Ctrl+C. In the cloud (module 9) the same flow runs on managed infrastructure instead.
"""

from __future__ import annotations

from retention.pipeline.flows import retrain

MONTHLY = "0 6 1 * *"


def main() -> None:
    retrain.serve(
        name="monthly-retrain",
        cron=MONTHLY,
        tags=["retention", "training"],
        description="Retrain on fresh data; promote only if the gate passes (criteria version 2).",
        parameters={"force": False, "auto_promote": True},
    )


if __name__ == "__main__":
    main()
