# Customer Retention ML Platform

[![CI](https://github.com/rafiulhc/customer-retention-ml-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/rafiulhc/customer-retention-ml-platform/actions/workflows/ci.yml)

Predicts which customers are likely to cancel, so a customer-success team can step in before they leave,
built end to end with production tooling: data validation, training, experiment tracking and a model
registry, pipelines, serving, CI, cloud deployment, and drift monitoring with retraining.

> Work in progress. Dataset: the public IBM Telco Customer Churn sample.

## Development

```bash
uv sync                          # create .venv with all dependencies
uv run pre-commit install        # run lint/format checks before every commit
uv run ruff check . && uv run mypy src && uv run pytest
```

## Experiment tracking (MLflow)

```bash
cp .env.example .env             # once: tracking URI and local settings (no secrets)
docker compose up -d --wait      # Postgres + SeaweedFS (S3) + MLflow + Prefect + the API
uv run python -m retention.tracking         # check the connection and the lineage tags
uv run python -m retention.models.train     # model comparison -> MLflow
uv run python -m retention.models.register  # train the chosen model, register a new version
uv run python -m retention.models.champion  # load @champion like serving does and verify it
uv run python -m retention.evaluation.evaluate  # one-time test-set evaluation -> MLflow + model version tags
```

MLflow UI: <http://localhost:5001>. Run data lives in Docker volumes; `docker compose down` keeps it.

The model registry holds `customer-retention-model`. Serving loads `models:/customer-retention-model@champion`.
The first version becomes `@champion` automatically; later versions need `--promote`.
Details: [docs/experiment_tracking.md](docs/experiment_tracking.md). Test-set results: [docs/evaluation.md](docs/evaluation.md).

## Retraining pipeline (Prefect)

```bash
uv run python -m retention.pipeline.flows   # fetch -> validate -> train -> verify -> evaluate -> gate -> promote
uv run python -m retention.pipeline.serve   # keep running: monthly schedule, runs from the UI
```

Prefect UI: <http://localhost:4200>. Details: [docs/pipeline.md](docs/pipeline.md).

## Serving

```bash
open http://localhost:8000/docs              # the online API (started by docker compose)
uv run python -m retention.pipeline.score    # this week's call list -> data/outputs/call_lists/
```

`POST /predict` returns churn risk, whether the customer is worth a call, the top reasons and the model version.
Details: [docs/serving.md](docs/serving.md).

## CI

Every push runs lint, strict type checks, 132 tests, a dependency vulnerability audit, a **model check**
(cross-validation against a committed baseline; never the test set) and a scan of the API image.
Details: [docs/ci.md](docs/ci.md).
