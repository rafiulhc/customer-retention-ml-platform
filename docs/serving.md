# Serving (module 7)

The champion is used in the two ways the framing doc asked for, and **both use the same scoring code**
(`serving/service.py`), so their numbers can't drift apart (no training/serving skew):

| | Online API | Weekly call list |
|---|---|---|
| For | one customer at a time (e.g. a CRM screen) | the success team's week |
| Runs | FastAPI container, always on | Prefect flow, Mondays 05:00 UTC |
| Input | JSON, validated against the data contract | the customer table, validated by `clean()` |
| Output | risk, call decision, top reasons, model version | top 100 by risk above the threshold, with reasons and expected value |

## The API

```bash
docker compose up -d --wait        # includes the api service
open http://localhost:8000/docs    # interactive documentation (OpenAPI)
```

| Endpoint | Returns |
|---|---|
| `POST /predict` | `p_leaves`, `worth_calling`, `reasons`, `model_version` for one customer |
| `POST /predict/batch` | the same for up to 1,000 customers, in input order |
| `GET /model` | the served version, its threshold and lineage: commit, data fingerprint, evaluation run, criteria, ship decision |
| `GET /health` | liveness, plus the model version |

Example (a new fiber customer on a monthly contract):

```json
{"customer_id": "9001-RISKY", "p_leaves": 0.7717, "worth_calling": true,
 "reasons": ["tenure = 2", "internet_service = Fiber optic", "contract = Month-to-month"], "model_version": "1"}
```

**The contract.** The request model is generated from `RAW_SCHEMA` (allowed values per field), so the API accepts
exactly what training accepted. Wrong values, missing fields and unknown fields (including `gender`, which the model
deliberately doesn't use) get a **422** that names the field and the allowed values.

**What gets served.** At startup the service resolves `@champion` to an **exact version** and loads three things
with it: the model, the call **threshold** chosen in that version's test evaluation (0.28 for v1), and the background
means used for explanations. It **refuses to start** for a version that was never evaluated (no threshold to decide
calls with) or when there is no champion. A server without a valid model must not take traffic. Every response
says which version produced it. To serve a newly promoted champion, restart the service (a rolling restart in
production).

**Reasons without SHAP.** For a linear model, SHAP values have a closed form: coefficient × (value − average). So
serving needs only about 40 background means, logged with each version, not the training data or the `shap`/`numba`
packages. A test checks the closed form equals `shap.LinearExplainer` exactly. Reasons are the features pushing
*towards* leaving, relative to an average customer. For low-risk customers they are small pushes, not causes for
concern.

**Measured.** p50 7.6 ms, **p95 8.9 ms** over HTTP to the container (requirement: p95 < 200 ms). Logs record method,
path, status and duration only: request bodies hold customer data and stay out of logs.

## The image

`infra/api/Dockerfile`: multi-stage, `uv sync --frozen --no-default-groups` (core dependencies only), base images
**pinned by digest**, runs as a **non-root user**, with a health check.

- **Dependencies are split by job** (`pyproject.toml`): *core* is what serving needs (`mlflow-skinny`, FastAPI,
  scikit-learn, pandas...). `training`, `pipeline` and `dev` are dependency groups. `uv sync` installs everything
  locally, and the image installs core only. None of xgboost, lightgbm, optuna, shap, numba, prefect, matplotlib or
  pytest is in the image, and a test checks that serving never imports them.
- **Size: 1.03 GB.** The environment is 619 MB (the full dev environment is 954 MB), almost all of it the scientific
  stack serving really uses: pyarrow 142 MB, scipy 149 MB (required by scikit-learn), pandas 75 MB, scikit-learn
  57 MB, numpy 68 MB.
- **Base image freshness vs the two-week rule.** Official Python images are rebuilt often for operating-system
  security patches (the pinned digest was built 6 days before it was pinned). Pinning by digest makes builds
  reproducible. Moving the digest forward is a deliberate, reviewed change. In CI (module 8) an image scanner decides
  when it must move.

## The weekly call list

`uv run python -m retention.pipeline.score` (scheduled by `uv run python -m retention.pipeline.serve`) writes
`data/outputs/call_lists/call_list_<date>.csv`:

```
rank,customer_id,p_leaves,expected_value,reasons,model_version,scored_at
1,5150-ITWWB,0.832,162.96,tenure = 3; internet_service = Fiber optic; contract = Month-to-month,1,...
```

First run: **5,174** active customers scored, **1,311** above the threshold, **100** on the list (capacity),
about **22** expected saves and **$13,834** expected value for the week. Expected value per call is
`p × save rate × value of saving − cost of a call`. It is meaningful because the probabilities are calibrated
(module 5). The Prefect summary shows counts only, never customer IDs.

*Dataset limitation:* the IBM sample is one snapshot, so "active" means `Churn = No` in it, and those customers
were in the training data. In production this step reads the current customer table.

## Bugs found while building this

- **A file without blank `TotalCharges` failed the data contract.** Left to guess, pandas read the column as numbers,
  and the contract (which expects text) rejected the file. The real data only worked because it happens to contain
  11 blanks. A month with no brand-new customers would have broken training and scoring. `read_raw()` now states the
  column types explicitly, everywhere the CSV is read, with a regression test.
- **The build backend's minimum version was 4 days old** (`uv_build>=0.12.23`, set when the project was created). The
  dependency cooldown caught it in the Docker build. The minimum is now the uv version pinned in the Dockerfile.
