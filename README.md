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
