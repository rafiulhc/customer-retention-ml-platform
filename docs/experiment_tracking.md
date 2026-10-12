# Experiment tracking and model registry

Every experiment and every model version is recorded in MLflow, with enough lineage to reproduce it.

## Architecture

```
code ──HTTP──▶ MLflow server ──▶ PostgreSQL     runs, params, metrics, tags, model registry
                  :5001       └─▶ S3 (SeaweedFS) artifacts: models, tables, trial logs
```

| | Local (`docker compose`) | AWS (module 9) |
|---|---|---|
| Tracking server | MLflow 3.16.1 container, `--serve-artifacts` | ECS service |
| Backend store | PostgreSQL 17 | RDS PostgreSQL |
| Artifact store | SeaweedFS 4.47 (S3 API) | S3 |

Clients only talk to the server (`--serve-artifacts`), so they never need storage credentials.
SeaweedFS replaces MinIO, whose open-source edition was archived in 2026.

## What gets logged

| Run | Parent holds | Each child holds |
|---|---|---|
| `model selection` (`retention.models.train`) | best model, its paired difference vs the B2 rule, `cv_summary.txt`, `cv_per_fold.csv` | one model: hyperparameters, `pr_auc` etc. as mean ± std, `fold_*` per fold (step = fold), `rank`, `is_final` |
| `tuning` (`retention.models.tune`) | `tuning.json` | one Optuna search (best params, `trials.csv`), plus the fresh-folds re-check per contender |
| `champion candidate` (`retention.models.register`) | the trained model, registered as a new version | |

**Lineage tags on every run:** `code.git_commit`, `code.git_dirty`, `data.raw_sha256`, `data.train_sha256`,
`data.test_sha256`, and row counts. Only runs from clean commits (`code.git_dirty = false`) count as official.

## Model registry

- **Name:** `customer-retention-model`. **Consumers load** `models:/customer-retention-model@champion`, never a file path.
- **Signature:** 17 named, typed feature columns in; `[P(stays), P(leaves)]` out (`predict_proba`). Requests that
  don't match are rejected with a schema error instead of producing a silently wrong prediction.
- **Promotion:** the first version becomes `@champion` automatically. Later versions need `--promote`
  (from module 8, the CI model gate decides). Rolling back means moving the alias, not redeploying.
- **Requirements:** pinned explicitly to the training environment. MLflow's automatic inference recorded
  whatever happened to be imported, `pytest` included.
- **Serialization:** skops, not pickle (loading a pickle can run arbitrary code).

## Checking the champion

`uv run python -m retention.models.champion` loads `@champion` from the server the way serving will, and checks:
the signature, the output shape, valid probabilities, determinism, rejection of a malformed request, and that a
fresh local retrain on the same data predicts identically. It exits non-zero on any failure, so CI can use it.

## Local gotchas

- **Port 5001**, because macOS uses 5000 for AirPlay Receiver.
- **`--allowed-hosts` includes the port** (`localhost:*`). MLflow 3 rejects other Host headers (403) to block DNS rebinding.
- **Presigned URLs:** MLflow 3.16 sends clients direct-to-S3 download URLs, which locally point at the Docker-only
  hostname `seaweedfs`. `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` (in `.env.example`) streams through the server instead.
- **SeaweedFS volumes:** each bucket is its own collection, so don't cap `-volume.max` at a small number.
