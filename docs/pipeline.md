# Retraining pipeline (module 6)

One Prefect flow, `retrain`, turns the separate steps of modules 2–5 into a single automated process
that is scheduled, observable and safe to re-run.

```
fetch (3 retries) → prepare & validate → new data? ──no──▶ skipped
                                            │ yes (or --force)
   compare models → register challenger → verify → evaluate once on test → gate ──pass──▶ promote @champion
                                                                              └─fail──▶ keep current champion
```

| Tool | Records |
|---|---|
| **Prefect** (http://localhost:4200) | *what ran and how it went*: each flow and task run, state, logs, duration, retries, the summary artifact |
| **MLflow** (http://localhost:5001) | *what was produced*: the model comparison, the challenger and its test evaluation, the registry |

## Steps

| Task | What it does | Fails the run when |
|---|---|---|
| `fetch_data` | Downloads the raw CSV, checksum-verified. **Retries 3 times** (10 s, 30 s, 60 s) | the network stays down, or the checksum is wrong |
| `prepare_data` | Validates (pandera), cleans, splits, writes the manifest | the data breaks the schema |
| `needs_training` | Compares the new training data with the champion's (content fingerprint) | — |
| `compare_models` | Re-runs the 6-model comparison; **warns** if the chosen family is no longer ranked first | — |
| `register_challenger` | Trains the chosen family, registers a new version **without promoting it** | — |
| `verify_challenger` | The serving contract checks from module 4.5 | the artifact is broken (never evaluated, never promoted) |
| `evaluate_challenger` | The challenger's one look at the test set (module 5) | — |
| `decide` | The promotion gate (criteria version 2) | — |
| `promote` | Moves `@champion`, tags the version with the gate's reasons | — |

**Promotion gate** (`evaluation/gate.py`, pre-registered in [problem_framing.md](problem_framing.md)):
the challenger must meet the shipping criteria against the B2 rule, **and** must not be worse than the current
champion on PR-AUC (95% interval lower bound above −0.01). "Not worse", not "better": a model retrained on fresh
data should replace a stale one even when it only ties. In the pipeline, **only the gate moves `@champion`**, even
for the first version. If no model passes, there is no champion, and the B2 rule is what ships.

## Running it

```bash
uv run python -m retention.pipeline.flows                # once; skips if the data is unchanged
uv run python -m retention.pipeline.flows --force        # retrain anyway
uv run python -m retention.pipeline.flows --no-promote   # register and evaluate, never promote
uv run python -m retention.pipeline.serve                # stays running: monthly schedule + UI-triggered runs
```

`serve` registers the deployment **monthly-retrain** (cron `0 6 1 * *`, 06:00 UTC on the 1st, per the framing
doc's "retrain monthly") and runs whatever the server schedules, while the process is running. Runs can also be
started from the Prefect UI. In the cloud (module 9) the same flow runs on managed infrastructure instead of a
terminal. Drift-triggered retraining comes in module 10.

## Incident: the first real run retrained on unchanged data

**What happened.** The first run decided the data had changed, trained **v3**, and the gate promoted it over v1.
v3 predicted **identically** to v1 (maximum difference 0.0), so production behaviour was unaffected.

**Why.** "Has the data changed?" compared **file checksums**. `train.parquet` had been rewritten by a newer
`prepare` run, with pyarrow 25.0.1 instead of the 26.0.0 used in module 2. Parquet files record the writer's
version, so **the bytes changed while the data didn't**.

**Fix.** A **content fingerprint** (`lineage.content_sha256`: a hash of the column names, types and values) is
now stored in the manifest and on each model version, and used for every "same data?" decision: retraining,
evaluation and the champion check. File checksums are kept for what they are good at: detecting corruption or
tampering. A regression test writes the same rows with different compression and expects the pipeline to skip.
`@champion` was restored to **v1** (the shipped decision). v3 is tagged with a note, and the fingerprint was
backfilled onto v1–v3 (justified by v1 and v3 predicting identically).

**Lesson.** Decide on *meaning*, not representation. A file's checksum answers "is this the same file?", not
"is this the same data?".

## Resource use (16 GB laptop)

The whole local stack (Postgres, SeaweedFS, MLflow, Prefect) uses about **1 GB**. MLflow alone used 2.2 GB at
first: its GenAI background job system (7 queue workers) is now off (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`),
and the web server runs 2 workers instead of 4.
