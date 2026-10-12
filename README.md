# Customer Retention ML Platform

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
docker compose up -d --wait      # Postgres + SeaweedFS (S3) + MLflow server
uv run python -m retention.tracking         # check the connection and the lineage tags
uv run python -m retention.models.train     # model comparison -> MLflow
uv run python -m retention.models.register  # train the chosen model, register a new version
uv run python -m retention.models.champion  # load @champion like serving does and verify it
```

MLflow UI: <http://localhost:5001>. Run data lives in Docker volumes; `docker compose down` keeps it.

The model registry holds `customer-retention-model`. Serving loads `models:/customer-retention-model@champion`.
The first version becomes `@champion` automatically; later versions need `--promote`.
Details: [docs/experiment_tracking.md](docs/experiment_tracking.md).
